"""Advanced training script adapted from CleanRL's repository: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py.

This is a full training script including CLI, logging and integration with TensorBoard and WandB for experiment tracking.

Full documentation and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy.

Note: default value for total-timesteps has been changed from 2 million to 8000, for easier testing.

Authors: Costa (https://github.com/vwxyzjn), Elliot (https://github.com/elliottower)

Modified to train on multi-car-racing environment by: Juan (https://github.com/JuanJoseZapata)
"""

# flake8: noqa

import argparse
import importlib
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool
import json

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_start_method('spawn', force=True)

from gym_multi_car_racing import multi_car_racing, multi_car_racing_bezier
from vector.vector_constructors import concat_vec_envs
from collections import deque
import scipy.stats as stats


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=3,
        help="seed of the experiment")
    parser.add_argument("--num-workers", type=int, default=16,
        help="Number of parallel workers for collecting rollouts")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="multi_car_racing",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="multi_car_racing",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_050_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-agents", type=int, default=1,
        help="number of agents in the environment")
    parser.add_argument("--penalties", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to add additional penalties to the environment")
    parser.add_argument("--penalty-weight", type=float, default=0.05,
        help="weight of the penalties")
    parser.add_argument("--frame-stack", type=int, default=4,
        help="number of stacked frames")
    parser.add_argument("--frame-skip", type=int, default=4,
        help="number of frames to skip (repeat action)")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=125,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--clip-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to clip rewards or not")
    parser.add_argument("--norm-rew", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to normalize rewards or not")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
        help="Adam optimizer weight decay")
    parser.add_argument("--discrete-actions", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Whether to use a discrete action space")
    parser.add_argument("--trained-agent", type=str, default=None,
        help="file name of an already trained agent that will be further trained")
    parser.add_argument("--bezier", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Whether to use bezier curves for the track")
    parser.add_argument("--curriculum", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Whether to use curriculum learning")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=False, ret=True, clipob=3., cliprew=3., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, truncations, infos = self.env.step(action)
        for agent_id in range(len(infos)):
            infos[agent_id]['episode']['real_reward'] = rews[agent_id]
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        self.ret = self.ret * (1-np.array(dones).astype(float))

        return obs, rews, dones, truncations, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self, seed=None, options=None):
        self.ret = np.zeros(())
        obs, infos = self.env.reset()
        return self._obfilt(obs), infos


from VAE.CarRacing.VAE import VAE

input_dim = 12 * 2 + 1
hidden_dim = 256
latent_dim = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load VAE model
vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device) # GPU
vae_model.load_state_dict(torch.load(f"scripts/VAE/CarRacing/models/vae_points_h={hidden_dim}_z={latent_dim}.pt"))
vae_model.eval()


def make_env():

    # env setup
    if args.bezier:
        env = multi_car_racing_bezier.parallel_env(n_agents=args.num_agents, use_random_direction=True,
                                render_mode="state_pixels", penalties=args.penalties,
                                penalty_weight=args.penalty_weight, use_ego_color=True,
                                discrete_action_space=args.discrete_actions)
    else:
        env = multi_car_racing.parallel_env(n_agents=args.num_agents, use_random_direction=True,
                                render_mode="state_pixels", penalties=args.penalties, use_ego_color=True,
                                discrete_action_space=args.discrete_actions)
    if not args.discrete_actions:
        env = ss.clip_actions_v0(env)
    if args.frame_skip > 1:
        env = ss.frame_skip_v0(env, args.frame_skip)
    if args.frame_stack > 1:
        env = ss.frame_stack_v1(env, args.frame_stack)
    if args.clip_rewards:
        env.render_mode = None
        env = ss.clip_reward_v0(env, lower_bound=-3, upper_bound=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    if args.norm_rew:
        env = NormalizedEnv(env)

    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(12, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        if args.discrete_actions:
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
        else:
            self.actor_mean = layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
            self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        if args.discrete_actions:
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
        else:
            action_mean = self.actor_mean(hidden)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)


if __name__ == "__main__":
    # --num-steps 32 --num-envs 6 --total-timesteps 256
    from datetime import datetime

    args = parse_args()
    print(args)
    method = "CL" if args.curriculum else "DR"
    run_name = f"{args.env_id}__{args.seed}__{datetime.now().strftime('%Y%m%d')}_{method}"

    # Save json file with hyperparameters
    with open(f"log/args/{run_name}.json", "w") as outfile:
        json.dump(vars(args), outfile)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    def map_envs_to_pipes(idx_starts, num_envs):
        num_pipes = len(idx_starts) - 1
        envs_per_pipe = int(np.ceil(num_envs / num_pipes))

        envs_pipes = np.ones((num_pipes, envs_per_pipe))*(-1)

        for i in range(envs_pipes.shape[0]):
            envs_pipes[i, :] = np.arange(idx_starts[i], idx_starts[i+1])
        
        return envs_pipes

    def get_env_position(env_num, envs_pipes):
        return [e[0] for e in np.where(envs_pipes == env_num)]

    def set_control_points(envs, env_num, control_points):
        envs_pipes = map_envs_to_pipes(envs.idx_starts, envs.num_envs)
        pipe, pos = get_env_position(env_num, envs_pipes)
        envs.pipes[pipe].send(("set_control_points", (pos, control_points)))
        envs.pipes[pipe].recv()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device:", device)

    # env setup
    envs = concat_vec_envs(make_env, args.num_envs // args.num_agents, num_cpus=args.num_workers)
    print("Observation space:", envs.observation_space)
    print("Action space:", envs.action_space)
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    agent = Agent(envs).to(device)
    if args.trained_agent is not None:
        agent.load_state_dict(torch.load(os.path.join("log/ppo", args.trained_agent)))
        print("Loaded trained agent")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=args.weight_decay)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    N = 100
    running_reward = deque([0 for _ in range(N)], maxlen=N)
    min_difficulty = 0
    max_difficulty = 11
    difficulties = np.arange(min_difficulty, max_difficulty, 1)
    difficulty = min_difficulty + 1  # Initial difficulty
    d = difficulty
    # Uniform weights
    weights = np.ones(difficulties.shape[0]) / difficulties.shape[0]
    # Add a cooldown variable to avoid changing the difficulty too often
    cooldown = 0

    # Load dataset with control points and difficulties
    X = np.load("scripts/VAE/CarRacing/X_30k_new.npy")
    Y = np.load("scripts/VAE/CarRacing/complexities_30k_new.npy")
    Y = Y/np.max(Y)*10

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_termination = torch.zeros(args.num_envs).to(device)
    next_truncation = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    prev_info = {}

    for update in range(1, num_updates + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncation, info = envs.step(
                action.cpu().numpy()
            )
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_termination, next_truncation = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(termination).to(device),
                torch.Tensor(truncation).to(device),
            )

            # TODO: fix this
            for idx in range(args.num_envs):
                player_idx = idx % 2 if args.num_agents == 2 else 0
                if next_termination[idx]:
                    running_reward.append(prev_info[idx]['episode']['r'])
                    
                    print(
                        f"global_step={global_step}, {player_idx}-episodic_return={prev_info[idx]['episode']['r']}, {player_idx}-episodic_length={prev_info[idx]['episode']['l']}, difficulty={d}, avg_return({N})={np.mean(running_reward)}"
                    )
                    writer.add_scalar(
                        f"charts/episodic_return-player{player_idx}",
                        prev_info[idx]["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/average_episodic_return-player{player_idx}",
                        np.mean(running_reward),
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/std_episodic_return-player{player_idx}",
                        np.std(running_reward),
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/difficulty-player{player_idx}",
                        difficulty,
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_length-player{player_idx}",
                        prev_info[idx]["episode"]["l"],
                        global_step,
                    )

                    if args.curriculum:
                        # Increase difficulty if the running reward is greater than 600
                        if np.mean(running_reward) > 550 and cooldown == 0:
                            difficulty += 1
                            cooldown = 2*N
                        # Decrease difficulty if the running reward is less than 300
                        elif np.mean(running_reward) < 300 and cooldown == 0:
                            difficulty -= 1
                            cooldown = 2*N
                        
                        # Decrease cooldown
                        cooldown = max(0, cooldown-1)

                        # Make sure the difficulty is within the bounds
                        difficulty = max(min_difficulty+1, min(max_difficulty, difficulty))
                        difficulties = np.arange(min_difficulty, difficulty, 1)

                        # Calculate weights for the difficulty
                        weights = np.ones(difficulties.shape[0])  # Uniform distribution
                        #weights = stats.expon.pdf(difficulties, scale=4)[::-1]  # Exponential distribution
                        weights /= weights.sum()  # Make sum to 1

                        d = np.random.choice(difficulties, p=weights)

                        # With VAE
                        # z = np.random.uniform(-2, 2, size=(1, latent_dim))
                        # z = np.append(z, d)
                        # z = torch.tensor(z).to(device).ravel().float()
                        # # Reconstruct image using VAE
                        # control_points = vae_model.decoder(z).to('cpu').detach().numpy().squeeze().reshape(12,2) * 30  # Rescale

                        # Without VAE (use dataset)
                        idxs = np.where(np.abs(Y-d) < 1)[0]
                        control_points = X[np.random.choice(idxs)]

                        # Set new control points
                        set_control_points(envs, idx, control_points)

                    else:
                        set_control_points(envs, idx, None)

            prev_info = info

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            next_done = torch.maximum(next_termination, next_truncation)
            dones = torch.maximum(terminations, truncations)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.discrete_actions:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        # Save model checkpoint
        checkpoint_interval = num_updates // 100
        if update % checkpoint_interval == 0:
            torch.save(agent.state_dict(), f"log/ppo/{run_name}_{global_step}.pt")

    envs.close()
    writer.close()