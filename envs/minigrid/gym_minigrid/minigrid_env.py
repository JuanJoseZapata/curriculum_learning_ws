# Modified version of the tutorial https://minigrid.farama.org/content/create_env_tutorial/

from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from .minigrid_levels import get_minigrid_level
import random
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grid(z):
    """Get grid from a 2D numpy array with (x,y) coordinates"""

    points = np.copy(z)
    points = np.clip(points, 0, 12)  # 15x15 grid - 2 for outer walls

    grid = np.zeros((13, 13), dtype=int)

    # Place blocks (except for the last two points, which are the player and goal)
    for i in range(points.shape[0] - 2):
        grid[points[i, 0], points[i, 1]] = 1

    occupied_cells = list(np.argwhere(grid == 1))
    occupied_cells = [tuple(cell) for cell in occupied_cells]

    player_pos = points[-2]
    goal_pos = points[-1]

    if tuple(player_pos) in occupied_cells:
        # Delete player position from occupied cells
        grid[player_pos[0], player_pos[1]] = 0
    
    if tuple(goal_pos) in occupied_cells:
        # Delete goal position from occupied cells
        grid[goal_pos[0], goal_pos[1]] = 0

    player_pos = np.array([player_pos[1], player_pos[0]])
    goal_pos = np.array([goal_pos[1], goal_pos[0]])

    return grid, player_pos, goal_pos


class Env(MiniGridEnv):
    def __init__(
        self,
        size=15,
        max_steps: int | None = None,
        num_tiles: int = 25,
        level: str | None = None,
        vae=None,
        agent_start_pos: list | None = None,
        agent_start_dir: int | None = None,
        goal_pos: list | None = None,
        bit_map: np.ndarray | None = None,
        training: bool = True,
        **kwargs,
    ):
        self.level = level
        self.num_tiles = num_tiles
        self.num_envs = 1
        self.agent_start_pos = np.array(agent_start_pos) if agent_start_pos is not None else None
        self.agent_start_dir = agent_start_dir
        self.goal_pos = np.array(goal_pos) if goal_pos is not None else None
        self.bit_map = bit_map
        self.size = size
        self.blocks = []
        self.difficulty = None
        self.vae = vae
        self.training = training

        if self.level is not None:
            self.bit_map, self.agent_start_pos, self.goal_pos = get_minigrid_level(self.level)
        if self.bit_map is not None:
            self.size = self.bit_map.shape[0] + 2          

        self.width = self.size
        self.height = self.size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * self.size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _rand_pos(self):
        for i in range(10_000):
            x, y = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
            if (x, y) not in self.blocks:
                return np.array([x, y])
        raise ValueError("Failed to find an empty cell")
        
    def place_goal(self, position=None):
        if self.goal_pos is None:
            while True:
                self.goal_pos = self._rand_pos()
                if (self.goal_pos != self.agent_start_pos).all():
                    break
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

    def place_agent(self):
        if self.agent_start_pos is None:
            while True:
                self.agent_start_pos = self._rand_pos()
                if (self.agent_start_pos != self.goal_pos).all():
                    break
        
        self.agent_start_dir = random.randint(0, 3)
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def _gen_grid(self, width=None, height=None):
        # Create an empty grid
        self.grid = Grid(self.size, self.size)

        self.blocks = []

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.size, self.size)

        # Place blocks if bit_map is provided
        if self.bit_map is not None:
            for x in range(self.bit_map.shape[0]):
                for y in range(self.bit_map.shape[1]):
                    if self.bit_map[y, x]:
                        self.blocks.append((x+1, y+1))
                        # Add an offset of 1 for the outer walls
                        self.grid.set(x+1, y+1, Wall())
        else:
            # Generate N random blocks
            for _ in range(self.num_tiles):
                # Ensure that the block doesn't overlap with the agent or goal
                agent_start_pos = np.array([self.agent_start_pos]).flatten()
                goal_pos = np.array([self.goal_pos]).flatten()
                while True:
                    x, y = self._rand_pos()
                    if (x, y) != tuple(agent_start_pos) and (x, y) != tuple(goal_pos):
                        break
                self.blocks.append((x, y))
                self.grid.set(x, y, Wall())

            # Create bit map
            self.bit_map = np.zeros((self.size, self.size), dtype=int)
            for x, y in self.blocks:
                self.bit_map[y, x] = 1
            self.bit_map = self.bit_map[1:-1, 1:-1]

        # Place agent and goal
        self.place_agent()
        self.place_goal()

        self.mission = "grand mission"

    def set_difficulty(self, difficulties, weights):
        self.difficulty = np.random.choice(difficulties, p=weights)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if self.difficulty is None:
            self.difficulty = np.random.choice([d for d in range(2,25)])

        # Reinitialize episode-specific variables
        if self.level is None and not self.training:
            self.agent_pos = None
            self.agent_dir = None
            self.goal_pos = None
            self.agent_start_dir = None
            self.agent_start_pos = None
            self.bit_map = None
        elif self.level is not None:
            self.bit_map, self.agent_start_pos, self.goal_pos = get_minigrid_level(self.level)

        # Generate grid from VAE
        if self.vae is not None and self.level is None:
            latent_dim = 24
            z = np.random.uniform(-2, 2, (1, latent_dim))
            z = np.append(z, self.difficulty)
            z = torch.tensor(z).to(device).ravel().float()
            # Reconstruct image using VAE
            recon_x = self.vae.decoder(z).to('cpu').detach().numpy().squeeze().reshape(-1,2).astype(np.int32)

            self.bit_map, self.agent_start_pos, self.goal_pos = get_grid(recon_x)
            # Add 1 to account for the outer walls
            self.agent_start_pos += 1
            self.goal_pos += 1

        # Generate a new random grid at the start of each episode
        self._gen_grid()

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True
            terminated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        info = {'episode': {'r': reward, 'l': self.step_count, 't': terminated, 'truncated': truncated, 'd': self.difficulty}}

        return obs, reward, terminated, truncated, info


def main():
    env = Env(render_mode="human", num_tiles=40, max_steps=50, training=True)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()