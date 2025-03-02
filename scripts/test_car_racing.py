import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym_multi_car_racing import multi_car_racing, multi_car_racing_f1, multi_car_racing_bezier
from vector.vector_constructors import concat_vec_envs

import scipy.stats as stats


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="multi_car_racing",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=123,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-episodes", type=int, default=10,
        help="number of episodes to test")
    parser.add_argument("--model-path", type=str, default=None,
        help="path to the model to be tested")
    parser.add_argument("--track-name", type=str, default=None,
        help="Formula 1 track name")
    parser.add_argument("--num-agents", type=int, default=2,
        help="number of agents in the environment")
    parser.add_argument("--frame-stack", type=int, default=4,
        help="number of stacked frames")
    parser.add_argument("--frame-skip", type=float, default=4,
        help="number of frames to skip (repeat action)")
    parser.add_argument("--num-envs", type=int, default=1,  # 16
        help="the number of parallel game environments")
    parser.add_argument("--discrete-actions", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Whether to use a discrete action space")
    parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Whether to render the environment")
    parser.add_argument("--bezier", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Whether to use bezier curves for the track")
    args = parser.parse_args()
    # fmt: on
    return args


from VAE.CarRacing.VAE import VAE

input_dim = 12 * 2 + 1
hidden_dim = 256
latent_dim = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load VAE model
vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device) # GPU
vae_model.load_state_dict(torch.load(f"scripts/VAE/CarRacing/models/vae_points_h={hidden_dim}_z={latent_dim}.pt"))
vae_model.eval()
vae_model = None


def make_env():

    render_mode = "human" if args.render else "state_pixels"

    # env setup
    if args.track_name is None and args.bezier:
        print("Bezier tracks")
        env = multi_car_racing_bezier.parallel_env(n_agents=args.num_agents, render_mode=render_mode,
                                                   use_random_direction=True,
                                                   discrete_action_space=args.discrete_actions, verbose=1)
    elif args.track_name is None and not args.bezier:
        print("Default tracks")
        env = multi_car_racing.parallel_env(n_agents=args.num_agents, use_random_direction=True, use_ego_color=True,
                                            discrete_action_space=args.discrete_actions, render_mode=render_mode)
    else:
        print("F1 track:", args.track_name)
        env = multi_car_racing_f1.parallel_env(n_agents=args.num_agents, use_random_direction=False,
                                render_mode=render_mode, discrete_action_space=args.discrete_actions,
                                track=args.track_name, verbose=1)
    
    if not args.discrete_actions:
        env = ss.clip_actions_v0(env)
    if args.frame_skip > 1:
        env = ss.frame_skip_v0(env, args.frame_skip)
    if args.frame_stack > 1:
        env = ss.frame_stack_v1(env, args.frame_stack)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

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


def set_control_points(envs, control_points):
    envs.unwrapped.vec_envs[0].par_env.unwrapped.set_control_points(control_points)


if __name__ == "__main__":

    args = parse_args()

    args.model_path = "log/ppo/multi_car_racing__3__20240325_CL_990000.pt"
    args.num_agents = 1
    args.track_name = None
    args.render = True if args.track_name is None else False
    args.num_episodes = 10
    difficulty = 0
    args.bezier = True
    args.seed = 2

    print(args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device:", device)

    # env setup
    envs = concat_vec_envs(make_env, 1)
    print("Observation space:", envs.observation_space)
    print("Action space:", envs.action_space)
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path))

    min_difficulty = 0
    max_difficulty = 11
    difficulties = np.arange(min_difficulty, max_difficulty, 1)
    difficulty += 1
    # Uniform weights
    weights = np.ones(difficulties.shape[0]) / difficulties.shape[0]

    # Run
    returns = []
    lengths = []
    for episode in range(args.num_episodes):
        next_obs, info = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        episode_return = np.zeros(args.num_agents)
        episode_length = np.zeros(args.num_agents)
        done = np.zeros(args.num_envs)
        while not done.all():
            action, _, _, _ = agent.get_action_and_value(next_obs)
            next_obs, reward, done, truncation, info = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            #print(f"Reward: {reward}")
            episode_return += reward.ravel()
            episode_length += 1
        print(f"Episode {episode}: return={episode_return}, length={episode_length}\n")
        returns.append(episode_return)
        lengths.append(episode_length)

        if vae_model is not None and args.track_name is None:
            # Make sure the difficulty is within the bounds
            difficulty = max(min_difficulty+1, min(max_difficulty, difficulty))
            difficulties = np.arange(min_difficulty, difficulty, 1)

            # Calculate weights for the difficulty
            # weights = np.ones(d-2)  # Uniform distribution
            weights = stats.expon.pdf(difficulties, scale=2)[::-1]  # Exponential distribution
            weights /= weights.sum()  # Make sum to 1

            d = np.random.choice(difficulties, p=weights)

            z = np.random.uniform(-2, 2, size=(1, latent_dim))
            z = np.append(z, d)
            z = torch.tensor(z).to(device).ravel().float()
            # Reconstruct image using VAE
            control_points = vae_model.decoder(z).to('cpu').detach().numpy().squeeze().reshape(12,2) * 30  # Rescale

            # Set new control points
            set_control_points(envs, control_points)
        elif vae_model is None and args.track_name is None:
            set_control_points(envs, None)
    
    if args.track_name is not None:
        print(f"Track: {args.track_name}")
    print(f"Return: {np.mean(returns)} (+/-{np.std(returns)})")
    print(f"Length: {np.mean(lengths)} (+/-{np.std(lengths)})")

