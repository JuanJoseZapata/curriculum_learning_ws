from gym_multi_car_racing import multi_car_racing

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import DummyVectorEnv
from tianshou.policy import MultiAgentPolicyManager, PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from torch.distributions import Independent, Normal

import torch
import os
import numpy as np
import supersuit as ss

from training import _get_env, _get_agents, _get_env_render
from network import DQN

# ======== Step 1: Environment setup =========
train_envs = DummyVectorEnv([_get_env_render for _ in range(1)])   # DummyVectorEnv
test_envs = DummyVectorEnv([_get_env_render for _ in range(1)])

# seed
# seed = 1626
# np.random.seed(seed)
# torch.manual_seed(seed)
# train_envs.seed(seed)
# test_envs.seed(seed)

# ======== Step 2: Agent setup =========
policy, optim, agents = _get_agents()

load_policy = False
# Load saved policy
if load_policy:
    for i, _ in enumerate(agents):
        policy.policies[f'car_{i}'].load_state_dict(torch.load(os.path.join("log", "ppo", "ppo_one-car_rgb_1-frame_ss_lr2e-4.pth")))#['model'])
        print("Loaded policy")
        
# ======== Step 3: Collector setup =========
buffer = VectorReplayBuffer(10_000, buffer_num=len(train_envs))

train_collector = Collector(
    policy,
    train_envs,
    buffer,
    exploration_noise=True,
)
test_collector = Collector(policy, test_envs, exploration_noise=False)

result = train_collector.collect(n_step=2500, random=False)  # batch size * training_num
