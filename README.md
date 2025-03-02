# Curriculum Learning for Deep Reinforcement Learning

Juan José Zapata González\
Karlsruhe Institute of Technology (KIT)\
Institute of Measurement and Control Systems (MRT)\
2024
___

This repository contains the implementation of my master's thesis on curriculum learning for deep multi-agent reinforcement learning.

## Getting Started

1. Install required packages from `Pipfile` using `Pipenv`:
```bash
pipenv install
```
2. Initiate `multi_car_racing` submodule:
```bash
git submodule init
git submodule update
```
3. Install `multi_car_racing` environment:
```bash
cd envs/multi_car_racing
pip install -e .
```
4. Install `minigrid` environment:
```bash
cd envs/minigrid
pip install -e .
```

## Usage

### Multi-Car Racing

![image](https://github.com/JuanJoseZapata/curriculum_learning_ws/assets/66514932/e0c20154-774a-474f-b2a3-4a60ce32c4f6)

This repository contains a variant of [`MultiCarRacing-v0`](https://github.com/igilitschenski/multi_car_racing), which is itself a variant of Gym's [`CarRacing-v0`](https://gym.openai.com/envs/CarRacing-v0/). The code was modified in order to be used with [`PettingZoo`](https://pettingzoo.farama.org/) (multi-agent environments) and [`CleanRL`](https://docs.cleanrl.dev/).

This environment is a multi-agent, 2D top-down racing game where the agents control cars and try to complete a track. The agents get a reward for each tile they visit and a penalty for driving on grass. Additionally, the agents receive a small negative reward for each time step. The episode ends when the agents complete the track or when 1000 steps are reached.

The environment has the following parameters:

| Parameter              | Type  | Description |
|------------------------| :---: |-------------|
| `n_agents`             |`int`  | Number of agents in environment (Default: `2`). |
| `direction`            |`str`  | Winding direction of the track. Can be `'CW'` or `'CCW'` (Default: `'CCW'`). |
| `use_random_direction` |`bool` | Randomize winding direction of the track. Disregards `direction` if enabled (Default: `True`). |
| `backwards_flag`       |`bool` | Shows a small flag if agent driving backwards (Default: `True`). |
| `h_ratio`              |`float`| Controls horizontal agent location in the state (Default: `0.25`) |
| `use_ego_color`        |`bool` | In each view the ego vehicle has the same color if  activated (Default: `False`). |
|`discrete_action_space` |`bool` | Whether to use a discrete action space (Default: `False`). |
|`grayscale`             |`bool` | Grayscale observations (84x84) (Default: `False`). |
|`percent_complete`      |`float`| Percentage of visited tiles necessary to terminate the episode (Default: `0.95`). |
|`domain_randomize`      |`bool` | Use random colors each time the environment is reset (Default: `False`). |
|`penalties`             |`bool` | Add additional penalties for driving on grass (Default: `False`). |

The observations in the original `CarRacing-v0` environment are RGB images of shape `(96x96x3)`. The images contain some indicators at the bottom (speed, sterring wheel position, etc.). In this implementation, the bottom indicators are removed and the observations have the shape `(84x84x3)` (or `(84x84)` if `grayscale` is `True`).

### Training
The training is done using the script in `train_car_racing.py`. The script can be run using the following command:
```bash
python scripts/train_car_racing.py
```
Training parameters can be set using command-line arguments, for example
```bash
python scripts/train_car_racing.py --total-timesteps 5000000 --num-agents 2
```
Available arguments for executing the training script:
| Parameter               | Type  | Description |
|-------------------------| :---: |-------------|
| `--exp-name`            | `str`   | Name of the experiment |
| `--seed`                | `int`   | Seed of the experiment. Default `1` |
| `--num-workers`        | `int`   | Number of parallel workers for collecting samples. Default `16` |
| `--torch-deterministic` | `bool`  | If toggled, `torch.backends.cudnn.deterministic=True`. Default `True` |
| `--cuda`                | `bool`  | If toggled, cuda will be enabled by default. Default `True` |
| `--track`               | `bool`  | Track with Weights and Biases. Default `True` |
| `--wandb-project-name`  | `str`   | wandb project name. Default `multi_car_racing` |
| `--env-id`              | `str`   | The id of the environment. Default `multi_car_racing` |
| `--total-timesteps`     | `int`   | Total timesteps of the experiments. Default `5000000` |
| `--learning-rate`       | `float` | Learning rate of the optimizer. Default `3e-4`    |
| `--num-agents`          | `int`   | Number of agents in the environment. Default `1` |
| `--penalties`           | `bool`  | Whether to add additional penalties to the environment. Default `False`    |
| `--penalty-weight`     | `float` | Weight of the additional penalties. Default `0.1` |
| `--frame-stack`         | `int`   | Number of stacked frames. Default `4` |
| `--frame-skip`          | `int`   | Number of frames to skip (repeat action). Default `4` |
| `--num-envs`            | `int`   | Number of parallel game environments. Default `8` |
| `--num-steps`           | `int`   | Number of steps to run in each environment per policy rollout. Default `250` |
| `--anneal-lr`           | `bool`  | Toggle learning rate annealing for policy and value networks. Default `False` |
| `--gamma`               | `float` | Discount factor gamma. Default `0.99` |
| `--gae-lambda`          | `float` | Lambda for the general advantage estimation. Default `0.95` |
| `--num-minibatches`     | `int`   | Number of num-minibatches. Default `4` |
| `--update-epochs`       | `int`   | The K epochs to update the policy. Default `4` |
| `--norm-adv`            | `bool`  | Toggles advantages normalization. Default `True` |
| `--clip-coef`           | `float` | Surrogate clipping coefficient. Default `0.2` |
| `--clip-vloss`          | `bool`  | Toggles whether or not to use a clipped loss for the value function. Default `True` |
| `--clip-rewards`       | `bool`  | Toggles whether or not to clip rewards. Default `True` |
| `--norm-rew`           | `bool`  | Toggles reward normalization. Default `True` |
| `--ent-coef`            | `float` | Coefficient of the entropy. Default `0.0` |
| `--vf-coef`             | `float` | Coefficient of the value function. Default `0.5` |
| `--max-grad-norm`       | `float` | The maximum norm for the gradient clipping. Default `0.5` |
| `--target-kl`           | `float` | The target KL divergence threshold. Default `None` |
| `--discrete-actions`    | `bool`  | Whether to use a discrete action space. Default `False` |
| `--trained-agent`      | `str`   | Path to a trained agent to be loaded. Default `None` |

### Testing
Evaluation is done similarly to training. The file `test_car_racing.py` contains the testing script, which can be run with the command:
 ```bash
python scripts/test_car_racing.py --model-path <model_path>
```
Available arguments for the testing script:
| Parameter               | Type  | Description |
|-------------------------| :---: |-------------|
| `--seed`                | `int`   | Seed of the experiment. Default `1` |
| `--torch-deterministic` | `bool`  | If toggled, `torch.backends.cudnn.deterministic=True`. Default `True` |
| `--cuda`                | `bool`  | If toggled, cuda will be enabled by default. Default `True` |
| `--num-episodes`        | `int`   | Number of episodes to test. Default `10` |
| `--model-path`          | `str`   | Path to the model to be tested |
| `--num-agents`          | `int`   | Number of agents in the environment. Default `1` |
| `--penalties`           | `bool`  | Whether to add additional penalties to the environment. Default `False`    |
| `--frame-stack`         | `int`   | Number of stacked frames. Default `4` |
| `--frame-skip`          | `int`   | Number of frames to skip (repeat action). Default `4` |
| `--num-envs`            | `int`   | Number of parallel game environments. Default `1` |
| `--discrete-actions`    | `bool`  | Whether to use a discrete action space. Default `False` |

## MiniGrid
![image](https://github.com/JuanJoseZapata/curriculum_learning_ws/assets/66514932/0f1936fc-b54c-4303-b636-a1a40fc97180)

The second environment used in this work is `MiniGrid`. It is a customized version of the original [`MiniGrid`](https://minigrid.farama.org/) environment from Farama Foundation.

This environment is a simple grid world environment with a 2D grid of tiles. The agent can move forward or change direction (left or right). The objective of the agent is to reach the goal tile in the smallest number of steps. The goal of the agent is to reach the goal tile while avoiding obstacles and collecting rewards. The environment is a single-agent environment.

The environment has the following parameters:

| Parameter              | Type  | Description |
|------------------------| :---: |-------------|
| `size`                 |`int`  | Size of the grid (Default: `15`). |
| `max-steps`            |`int`  | Maximum number of steps per episode (Default: `None`). |
| `num-tiles`            |`int`  | Number of tiles (blocks) in the grid (Default: `25`). |
| `level`                |`str`  | Level name. See available levels in `envs/minigrid/gym_minigrid/minigrid_levels.py` (Default: `None`). |
| `vae`                  |`None` | VAE model for generating levels (Default: `None`). |
| `agent-start-pos`      |`list` | Initial position of the agent (Default: `None`). |
| `agent-start-dir`      |`int`  | Initial direction of the agent (Default: `None`). |
| `goal-pos`             |`list` | Position of the goal tile (Default: `None`). |
| `bit-map`              |`np.ndarray` | Bit map of the grid (Default: `None`). |
| `training`             |`bool` | Whether the environment is used for training (Default: `True`). |

### Training
Agents can be trained using the script `train_minigrid.py`. The script can be run using the following command:
```bash
python scripts/train_minigrid.py
```
Training parameters can be set using command-line arguments, for example
```bash
python scripts/train_minigrid.py --total-timesteps 1000000
```

Arguments for the training script:
| Parameter               | Type  | Description |
|-------------------------| :---: |-------------|
| `--exp-name`            | `str`   | Name of the experiment |
| `--learning-rate`       | `float` | Learning rate of the optimizer. Default `1e-4`    |
| `--seed`                | `int`   | Seed of the experiment. Default `1` |
| `--total-timesteps`     | `int`   | Total timesteps of the experiments. Default `10_050_000` |
| `--torch-deterministic` | `bool`  | If toggled, `torch.backends.cudnn.deterministic=True`. Default `True` |
| `--cuda`                | `bool`  | If toggled, cuda will be enabled by default. Default `True` |
| `--track`               | `bool`  | Track with Weights and Biases. Default `True` |
| `--wandb-project-name`  | `str`   | wandb project name. Default `minigrid` |
| `--wandb-entity`        | `str`   | The entity (team) of wandb's project. Default `None` |
| `--env-id`              | `str`   | The id of the environment. Default `minigrid` |
| `--num-minibatches`     | `int`   | Number of num-minibatches. Default `4` |
| `--num-envs`            | `int`   | Number of parallel game environments. Default `32` |
| `--num-steps`           | `int`   | Number of steps to run in each environment per policy rollout. Default `128` |
| `--gamma`               | `float` | Discount factor gamma. Default `0.995` |
| `--gae-lambda`          | `float` | Lambda for the general advantage estimation. Default `0.95` |
| `--ent-coef`            | `float` | Coefficient of the entropy. Default `0.0` |
| `--vf-coef`             | `float` | Coefficient of the value function. Default `0.5` |
| `--max-grad-norm`       | `float` | The maximum norm for the gradient clipping. Default `0.5` |
| `--clip-coef`           | `float` | Surrogate clipping coefficient. Default `0.2` |
| `--update-epochs`       | `int`   | The K epochs to update the policy. Default `4` |
| `--kle-stop`            | `bool`  | If toggled, the policy updates will be early stopped w.r.t target-kl. Default `False` |
| `--kle-rollback`        | `bool`  | If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl. Default `False` |
| `--target-kl`           | `float` | The target KL divergence threshold. Default `None` |
| `--gae`                 | `bool`  | Use GAE for advantage computation. Default `True` |
| `--norm-adv`            | `bool`  | Toggles advantages normalization. Default `True` |
| `--anneal-lr`           | `bool`  | Toggle learning rate annealing for policy and value networks. Default `False` |
| `--clip-vloss`          | `bool`  | Toggles whether or not to use a clipped loss for the value function. Default `True` |
| `--curriculum`          | `bool`  | Whether to use curriculum learning. Default `True` |

### Testing
Similar to CarRacing, trained agents can be tested with the script `test_minigrid.py`. It can be run with this command:

 ```bash
python scripts/test_minigrid.py --model-path <model_path>
```
Available arguments for the testing script:
| Parameter               | Type  | Description |
|-------------------------| :---: |-------------|
| `--seed`                | `int`   | Seed of the experiment. Default `1` |
| `--torch-deterministic` | `bool`  | If toggled, `torch.backends.cudnn.deterministic=True`. Default `True` |
| `--cuda`                | `bool`  | If toggled, cuda will be enabled by default. Default `True` |
| `--model-path`          | `str`   | Path to the model to be tested |
| `--level-name`          | `str`   | Zero-shot level name |
| `--num-envs`            | `int`   | Number of parallel game environments. Default `1` |
| `--num-steps`           | `int`   | Number of steps to run in each environment per policy rollout. Default `128` |
| `--num-episodes`        | `int`   | Number of episodes to test. Default `10` |
