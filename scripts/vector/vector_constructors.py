import gymnasium.vector
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from constructors import MakeCPUAsyncConstructor

def vec_env_args(env_fn, num_envs):
    env = env_fn()
    env_fns = [env_fn] * num_envs

    return env_fns, env.observation_space, env.action_space


def concat_vec_envs(vec_env_fn, num_vec_envs, num_cpus=1):
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env_fn, num_vec_envs))

    return vec_env
