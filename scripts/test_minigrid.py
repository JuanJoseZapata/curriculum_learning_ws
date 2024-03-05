"""MiniGrid PPO training script adapted from: https://github.com/vwxyzjn/gym_minigrid/blob/master/ppo.py.

Author: Costa Huang (https://github.com/vwxyzjn)

Modified by: Juan (https://github.com/JuanJoseZapata)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from gym_minigrid import minigrid_env
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from gymnasium.core import ObservationWrapper
import supersuit as ss
import time
import random
import os
import scipy.stats as stats


def parse_args():
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MiniGrid-15x15",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2_000_000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if not args.seed:
        args.seed = int(time.time())

    return args

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env_obs = RGBImgObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
        >>> env_obs = RGBImgPartialObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![RGBImgPartialObsWrapper](../figures/lavacrossing_RGBImgPartialObsWrapper.png)
    """

    def __init__(self, env, tile_size=1):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

    def observation(self, obs):
        rgb_img_partial = self.unwrapped.get_frame(tile_size=self.tile_size, agent_pov=True)

        return rgb_img_partial


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 256)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(256, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


from VAE.MiniGrid.VAE import VAE

vae_path = 'scripts/VAE/MiniGrid/models/VAE_MiniGrid_latent-dim-24_25-blocks.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = (25 + 2)*2 + 1  # (25 blocks + 1 player position + 1 goal position) + 1 complexity
hidden_dim = 128
latent_dim = 24
vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device)
vae_model.load_state_dict(torch.load(vae_path))

def make_env():
        env = minigrid_env.Env(size=15, agent_view_size=5,
                               num_tiles=40, level="Maze",
                               vae=None,
                               render_mode=render_mode)
        env = RGBImgPartialObsWrapper(env)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        return env

def set_difficulty(env, difficulty, weights):
    env.set_difficulty(difficulty, weights)


render = True
render_mode = "human" if render else None

min_difficulty = 2
max_difficulty = 25
difficulties = np.arange(min_difficulty, max_difficulty, 1)
difficulty = 25


if __name__ == "__main__":

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Create environment
    env = make_env()
    agent = Agent(env).to(device)

    # Load trained agent
    agent.load_state_dict(torch.load('log/ppo/minigrid_15M_curriculum.pt'))

    # Test agent
    num_episodes = 100
    rewards = []

    for i in range(num_episodes):
        next_obs, _ = env.reset()
        next_obs = torch.Tensor(next_obs).to(device).unsqueeze(0)
        next_done = torch.zeros(args.num_envs).to(device)
        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        )

        done = False
        while not done:
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
            next_obs, reward, done, trunc, info = env.step(action.item())
            next_obs = torch.Tensor(next_obs).to(device).unsqueeze(0)
            env.render()
        print(f'Episode {i} finished with reward {reward}')
        rewards.append(reward)

        # Make sure the difficulty is within the bounds
        difficulty = max(min_difficulty, min(max_difficulty, difficulty))

        # Calculate weights for the difficulty
        weights = stats.norm.pdf(difficulties, difficulty, 2.5)
        weights += (1 - weights.sum())/weights.shape[0]  # Make sum to 1

        # Set the difficulty
        set_difficulty(envs, difficulties, weights)

    print(f'Average reward: {np.mean(rewards):.4f} +- {np.std(rewards):.4f}')
    
