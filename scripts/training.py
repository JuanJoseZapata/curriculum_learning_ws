from tianshou.env.pettingzoo_env import PettingZooEnv
from pettingzoo.utils import parallel_to_aec

from gym_multi_car_racing import multi_car_racing

import os
import gymnasium as gym
import numpy as np
import torch

import time

from typing import Optional, Tuple, List

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.optim.lr_scheduler import LambdaLR

from torch.distributions import Independent, Normal

import supersuit as ss

from network import DQN

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter

# Number of agents
n_agents = 1
# Number of steps per epoch
step_per_epoch = 10000
# Number of steps per collect
step_per_collect = 700
# Number of epochs
max_epoch = 500
# PPO parameters
max_grad_norm = 0.5
gamma = 0.99
eps_clip = 0.2
vf_coef = 0.5
ent_coef = 0.0
rew_norm = True
norm_adv = True
recompute_adv = 0
value_clip = True
gae_lambda = 0.95
# Initial learning rate
lr = 1e-4
# Train num
train_num = 10
# Test num
test_num = 10
# Frame stack
frame_stack = 4
# Frame skip
frame_skip = 0
# Penalties
penalties = False
# Domain randomization
domain_randomize = False

# Resume training
run_id = None
resume_from_log = False if run_id is None else True

# Policy name
policy_name_load = None
policy_name_save = "ppo_1-car_4-frames_lr1e-4_700-steps-per-collect"

def _get_train_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = multi_car_racing.env(n_agents=n_agents, use_random_direction=True,
                               render_mode="state_pixels", penalties=penalties,
                               domain_randomize=domain_randomize)
    if frame_skip > 1:
        env = ss.frame_skip_v0(env, frame_skip)
    if frame_stack > 1:
        env = ss.frame_stack_v1(env, frame_stack)
    return PettingZooEnv(env)

def _get_test_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = multi_car_racing.env(n_agents=n_agents, use_random_direction=True,
                               render_mode="state_pixels", domain_randomize=domain_randomize)
    if frame_skip > 1:
        env = ss.frame_skip_v0(env, frame_skip)
    if frame_stack > 1:
        env = ss.frame_stack_v1(env, frame_stack)
    return PettingZooEnv(env)

def _get_agents(
    agents: Optional[List[BasePolicy]] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_train_env()
    observation_space = env.observation_space['observation'] if isinstance(
    env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    device = "cuda"
    print("Observation space:", observation_space)
    print("Action space:", env.action_space)
    env.close()

    if agents is None:
        agents = []
        optims = []
        for _ in range(n_agents):
            # model
            net = DQN(
                observation_space.shape[2],
                observation_space.shape[1],
                observation_space.shape[0],
                device=device
            ).to(device)

            actor = ActorProb(
                net, action_shape, max_action=max_action, device=device
            ).to(device)
            net2 = DQN(
                observation_space.shape[2],
                observation_space.shape[1],
                observation_space.shape[0],
                device=device
            ).to(device)
            critic = Critic(net2, device=device).to(device)
            for m in set(actor.modules()).union(critic.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(
                set(actor.parameters()).union(critic.parameters()), lr=lr
            )

            def dist(*logits):
                return Independent(Normal(*logits), 1)
        
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(step_per_epoch / step_per_collect) * max_epoch

            lr_scheduler = None #LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
            #lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 0.99**epoch)

            agent = PPOPolicy(
                actor,
                critic,
                optim,
                dist,
                discount_factor=gamma,
                max_grad_norm=max_grad_norm,
                eps_clip=eps_clip,
                vf_coef=vf_coef,
                ent_coef=ent_coef,
                reward_normalization=rew_norm,
                advantage_normalization=norm_adv,
                recompute_advantage=recompute_adv,
                # dual_clip=args.dual_clip,
                # dual clip cause monotonically increasing log_std :)
                value_clip=value_clip,
                gae_lambda=gae_lambda,
                action_space=env.action_space,
                lr_scheduler=lr_scheduler,
                action_bound_method='clip'
            )
            agents.append(agent)
            optims.append(optim)

    policy = MultiAgentPolicyManager(
        agents, env, action_scaling=True, action_bound_method='clip'
    )
    return policy, optims, env.agents


if __name__ == "__main__":
    print("STARTING TRAINING")

    # env = _get_env()

    # ======== Step 1: Environment setup =========
    train_envs = SubprocVectorEnv([_get_train_env for _ in range(train_num)])   # DummyVectorEnv, SubprocVectorEnv
    test_envs = SubprocVectorEnv([_get_test_env for _ in range(test_num)])

    # seed
    # seed = 1626
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # train_envs.seed(seed)
    # test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optims, agents = _get_agents()

    # Load saved policy
    if policy_name_load is not None:
        for i, _ in enumerate(agents):
            try:
                policy.policies[f'car_{i}'].load_state_dict(torch.load(os.path.join("log", "ppo", f"{policy_name_load}.pth"))['model'])
            except KeyError:
                policy.policies[f'car_{i}'].load_state_dict(torch.load(os.path.join("log", "ppo", f"{policy_name_load}.pth")))
            #optims[i].load_state_dict(torch.load(os.path.join("log", "ppo", f"{policy_name_load}.pth"))['optim'])

    # ======== Step 3: Collector setup =========
    buffer = VectorReplayBuffer(100_000, buffer_num=len(train_envs))

    train_collector = Collector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    #train_collector.collect(n_step=64*10)  # batch size * training_num
    # ======== Step 4: Callback functions setup =========

    agent = 0  # learn_agent

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ppo", f"{policy_name_save}.pth")
        os.makedirs(os.path.join("log", "ppo"), exist_ok=True)
        torch.save(policy.policies[agents[agent]].state_dict(), model_save_path)
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        #ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.policies[agents[agent]].state_dict(),
                "optim": optims[0].state_dict(),
            }, ckpt_path
        )
        return ckpt_path

    def stop_fn(mean_rewards):
        return mean_rewards >= 900

    def reward_metric(rews):
        return rews[:, agent]


    log_path = os.path.join("log", "ppo")

    logger = WandbLogger(save_interval=1, project='multi_car_racing', name=f'{policy_name_save}', run_id=run_id)
    writer = SummaryWriter()
    logger.load(writer)

    print("CUDA available:", torch.cuda.is_available())

    # ======== Step 5: Run the trainer =========
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=1,
        episode_per_test=5,
        batch_size=32,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        resume_from_log=resume_from_log,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
        verbose=True
    )

    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")