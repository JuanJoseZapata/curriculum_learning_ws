# Curriculum Learning for Deep Reinforcement Learning

![image](https://github.com/JuanJoseZapata/curriculum_learning_ws/assets/66514932/e0c20154-774a-474f-b2a3-4a60ce32c4f6)

Juan José Zapata González\
Karlsruhe Institute of Technology (KIT)\
Institute of Measurement and Control Systems (MRT)\
2023
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
cd multi_car_racing
pip install -e .
```

## Usage

### Multi-Car Racing
This repository contains a variant of [`MultiCarRacing-v0`](https://github.com/igilitschenski/multi_car_racing), which is itself a variant of Gym's [`CarRacing-v0`](https://gym.openai.com/envs/CarRacing-v0/). The code was modified in order to be used with the [`PettingZoo`](https://pettingzoo.farama.org/) and [`Tianshou`](https://tianshou.readthedocs.io/en/master/) libraries.

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

The _scripts_ folder contains the following scripts:
- `network.py`: This is where the neural network used in training is defined.
- `training.py`: Training script.
- `test_multi_car_racing_env.py`: Script for testing trained agents in the multi-car racing environment.

### Training
to do
### Testing
to do
