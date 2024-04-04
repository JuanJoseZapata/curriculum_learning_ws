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
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MiniGrid-15x15",
                        help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument("--level-name", type=str, default=None,
        help="Zero-shot level name")

    # Algorithm specific arguments
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the number of steps per game environment')

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


class PartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
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
        return obs["image"]


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
            layer_init(nn.Conv2d(16, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(288, 512)),
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
        hidden = self.network(x.permute((0, 3, 1, 2)) / 8.)

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

# input_dim = (25 + 2)*2 + 1  # (25 blocks + 1 player position + 1 goal position) + 1 complexity
# hidden_dim = 128
# latent_dim = 24
# vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device)
# vae_model.load_state_dict(torch.load(vae_path))
vae_model = None

def make_env(args):
        env = minigrid_env.Env(size=15, agent_view_size=7,
                               num_tiles=40, level=args.level_name,
                               max_steps=250, vae=vae_model,
                               render_mode=render_mode)
        env = PartialObsWrapper(env)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        return env

def set_difficulty(env, difficulty, weights):
    env.set_difficulty(difficulty, weights)


def zero_shot_benchmark(args, levels, agent_name, method, num_episodes=10, save_csv=False, verbose=0):

    rewards_all = {level: np.zeros(num_episodes) for level in levels}

    for level_name in levels:
        args.level_name = level_name
        if verbose:
            print(f'Level: {level_name}')

        # Create environment
        env = make_env(args)
        agent = Agent(env).to(device)

        # Load trained agent
        
        agent.load_state_dict(torch.load(f'log/ppo/{agent_name}'))  #minigrid__1__20240315_CL_10027008

        # Test agent
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
            if verbose:
                print(f'Episode {i} finished with reward {reward}')
            rewards.append(reward)

        rewards_all[level_name] = rewards
        if verbose:
            print(level_name)
            print(f'Average reward: {np.mean(rewards):.4f} +- {np.std(rewards):.4f}\n')

    if save_csv:
        rewards_all = pd.DataFrame(rewards_all)
        rewards_all.to_csv(f'minigrid_{args.seed}_{method}.csv')
    
    return {level_name: np.mean(rewards) for level_name, rewards in rewards_all.items()}


render = False
render_mode = "human" if render else None

min_difficulty = 2
max_difficulty = 25
difficulties = np.arange(min_difficulty, max_difficulty, 1)
difficulty = 25

levels = ["Maze", "Maze2", "Labyrinth", "Labyrinth2", "SixteenRooms", "SixteenRooms2", "SmallCorridor", "LargeCorridor"]


if __name__ == "__main__":

    args = parse_args()

    agent_name = 'minigrid__1__20240316_CL_8552448.pt'
    method = 'CL'
    num_episodes = 20
    args.seed = 2
    
    # Seeding
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    zero_shot_benchmark(args, levels, agent_name, method, num_episodes, save_csv=True, verbose=1)   
