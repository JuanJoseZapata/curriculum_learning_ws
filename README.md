# Curriculum Learning for Deep RL

Juan José Zapata González

Karlsruhe Institute of Technology

Institute of Measurement and Control Systems (MRT)

Summer Semester 2023
___

This repository contains the implementation of my master's thesis on curriculum learning for multi-agent deep reinforcement learning.

## Getting Started

1. Install required packages from `Pipfile` using `Pipenv`:
```shell
pipenv install
```
2. Initiate `multi_car_racing` submodule:
```shell
git submodule init
git submodule update
```
3. Install `multi_car_racing` environment:
```shell
cd multi_car_racing
pip install -e .
```

## Usage

### Multi-Car Racing
This repository contains a variant of [`MultiCarRacing-v0`](https://github.com/igilitschenski/multi_car_racing), which is itself a variant of Gym's [`CarRacing-v0`](https://gym.openai.com/envs/CarRacing-v0/). The code was modified in order to be used with the [`PettingZoo`](https://pettingzoo.farama.org/) and [`Tianshou`](https://tianshou.readthedocs.io/en/master/) libraries.

The environment has the following parameters:

| Parameter              | Type  | Description |
|------------------------| :---: |-------------|
| `num_agents`           |`int`  | Number of agents in environment (Default: `2`) |
| `direction`            |`str`  | Winding direction of the track. Can be `'CW'` or `'CCW'` (Default: `'CCW'`)|
| `use_random_direction` |`bool` | Randomize winding direction of the track. Disregards `direction` if enabled (Default: `True`). |
| `backwards_flag`       |`bool` | Shows a small flag if agent driving backwards (Default: `True`). |
| `h_ratio`              |`float`| Controls horizontal agent location in the state (Default: `0.25`) |
| `use_ego_color`        |`bool` | In each view the ego vehicle has the same color if  activated (Default: `False`). |

The _scripts_ folder contains the following scripts:
- `network.py`: This is where the neural network used in training is defined.
- `training.py`: Training script.
- `test_multi_car_racing_env.py`: Script for testing trained agents in the multi-car racing environment.

### Training
to do
### Testing
to do