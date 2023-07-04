from gym_multi_car_racing import multi_car_racing

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import DummyVectorEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
from tianshou.data import Collector


n_agents = 2
# Step 1: Load the PettingZoo environment
env = multi_car_racing.env(n_agents=n_agents, direction="CCW", render_mode="human")

# Step 2: Wrap the environment for Tianshou interfacing
env = PettingZooEnv(env)

# Step 3: Define policies for each agent
policies = MultiAgentPolicyManager([RandomPolicy() for _ in range(n_agents)], env)

# Step 4: Convert the env to vector format
env = DummyVectorEnv([lambda: env])

# Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
collector = Collector(policies, env)

# Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
result = collector.collect(n_episode=1)

env.close()