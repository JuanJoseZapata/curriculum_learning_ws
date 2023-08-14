from gym_multi_car_racing import multi_car_racing, multi_car_racing_f1
import gymnasium as gym
from gym_multi_car_racing.formula1 import *

from typing import Optional, Tuple, List

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import DummyVectorEnv
from tianshou.policy import MultiAgentPolicyManager, PPOPolicy, BasePolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from torch.distributions import Independent, Normal

import torch
import os
import numpy as np
import supersuit as ss

from network import DQN

# Parameters
n_agents = 1
frame_stack = 4
render_mode = "human"  # "state_pixels" or "human"
f1_track = None  # None or Belgium, Monaco, ... (see formula1.py)
policy_name = "ppo_1-car_best-agent_original-env"


def _get_test_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    if f1_track is None:
        env = multi_car_racing.env(n_agents=n_agents,
                                   use_random_direction=True,
                                   render_mode=render_mode,
                                   verbose=True,
                                   percent_complete=0.99)
    else:
        env = multi_car_racing_f1.env(f1_track,
                                      n_agents=n_agents,
                                      use_random_direction=False,
                                      render_mode=render_mode,
                                      verbose=True,
                                      percent_complete=0.99)
        
    if frame_stack > 1:
        env = ss.frame_stack_v1(env, frame_stack)
    return PettingZooEnv(env)

def _get_agents(
    n_agents: Optional[int] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_test_env()
    observation_space = env.observation_space['observation'] if isinstance(
    env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    device = "cuda"
    print("Observation space:", observation_space)
    print("Action space:", env.action_space)
    env.close()

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
            set(actor.parameters()).union(critic.parameters()), lr=1e-4
        )

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        agent = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
        )
        agents.append(agent)
        optims.append(optim)

    policy = MultiAgentPolicyManager(
        agents, env, action_scaling=True, action_bound_method='clip'
    )
    return policy, optims, env.agents


if __name__ == "__main__":

    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_test_env for _ in range(1)])   # DummyVectorEnv

    # seed
    # seed = 1626
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # train_envs.seed(seed)
    # test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents(n_agents=n_agents)

    # Load saved policy
    if policy_name is not None:
        for i, _ in enumerate(agents):
            try:
                policy.policies[f'car_{i}'].load_state_dict(torch.load(os.path.join("log", "ppo", f"{policy_name}.pth"))['model'])
            except KeyError:
                policy.policies[f'car_{i}'].load_state_dict(torch.load(os.path.join("log", "ppo", f"{policy_name}.pth")))
            print("Loaded policy")
            
    # ======== Step 3: Collector setup =========
    buffer = VectorReplayBuffer(10_000, buffer_num=len(train_envs), stack_num=frame_stack)

    train_collector = Collector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True,
    )

    if f1_track is not None:
        print(f"Track: {f1_track.name}")
    result = train_collector.collect(n_episode=10, random=False)
    print(result)
