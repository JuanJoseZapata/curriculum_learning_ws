from tianshou.env.pettingzoo_env import PettingZooEnv

from gym_multi_car_racing import multi_car_racing

import os
import gymnasium as gym
import numpy as np
import torch

import time

from typing import Optional, Tuple

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
step_per_collect = 400
# Number of epochs
max_epoch = 50
# Max grad norm
max_grad_norm = 0.5
# Initial learning rate
lr = 2e-4
# Frame stack
frame_stack = 4

# Resume training
run_id = None

# Policy name
policy_name_load = None
policy_name_save = "ppo_one-car_rgb_4-frames_lr2e-4"

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = multi_car_racing.env(n_agents=n_agents, use_random_direction=True,
                               render_mode="state_pixels", discrete_action_space=False)
    if frame_stack > 1:
        env = ss.frame_stack_v1(env, frame_stack)
    return PettingZooEnv(env)

def _get_env_render():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = multi_car_racing.env(n_agents=n_agents, use_random_direction=True,
                               render_mode="human", discrete_action_space=False)
    if frame_stack > 1:
        env = ss.frame_stack_v1(env, frame_stack)
    return PettingZooEnv(env)

def dist(*logits):
        return Independent(Normal(*logits), 1)

def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    agents = []
    optims = []
    for agent_id in env.agents:
        print(agent_id)
        observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
        ) else env.observation_space
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]
        device = "cuda"
        print("Observation space:", observation_space)
        print("Action space:", env.action_space)

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
        
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(step_per_epoch / step_per_collect) * max_epoch

        lr_scheduler = None #LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

        agent = PPOPolicy(
                    actor,
                    critic,
                    optim,
                    dist,
                    discount_factor=0.99,
                    value_clip=True,
                    reward_normalization=True,
                    max_grad_norm=max_grad_norm,
                    action_space=env.action_space,
                    action_scaling=True,
                    lr_scheduler=lr_scheduler,
                    )
        
        agents.append(agent)
        optims.append(optim)

    policy = MultiAgentPolicyManager(
        agents, env, action_scaling=False, action_bound_method='clip'
    )
    return policy, optims, env.agents


if __name__ == "__main__":
    print("STARTING TRAINING")

    env = _get_env()

    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(1)])   # DummyVectorEnv
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])

    # seed
    seed = 42
    np.random.seed(seed)
    #torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optims, agents = _get_agents()

    # Load saved policy
    if policy_name_load is not None:
        for i, _ in enumerate(agents):
            policy.policies[f'car_{i}'].load_state_dict(torch.load(os.path.join("log", "ppo", f"{policy_name_load}.pth")))
        
    # ======== Step 3: Collector setup =========
    buffer = VectorReplayBuffer(10_000, buffer_num=len(train_envs), ignore_obs_next=True)

    train_collector = Collector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
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
        return mean_rewards >= 800

    def reward_metric(rews):
        return rews[:, agent]


    log_path = os.path.join("log", "ppo")

    logger = WandbLogger(save_interval=1, project='multi_car_racing', name=f'{policy_name_save}', run_id=run_id)
    writer = SummaryWriter()
    logger.load(writer)

    # ======== Step 5: Run the trainer =========
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=2,
        episode_per_test=6, ###############################################
        batch_size=64,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
        verbose=True
    )

    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")