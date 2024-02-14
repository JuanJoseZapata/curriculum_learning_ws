# Modified version of the tutorial https://minigrid.farama.org/content/create_env_tutorial/

from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import random


class Env(MiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        num_tiles: int = 25,
        **kwargs,
    ):
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.num_tiles = num_tiles
        self.num_envs = 1
        self.size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _rand_pos(self):
        x, y = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
        return x, y

    def place_goal(self, position=None):
        if position is not None:
            x_goal, y_goal = position
        else:
            x_goal, y_goal = self._rand_pos()
        self.put_obj(Goal(), x_goal, y_goal)
        self.goal_pos = (x_goal, y_goal)

    def place_agent(self, position=None, direction=None):
        if position is not None:
            x_agent, y_agent = position
        else:
            while True:
                self.agent_start_pos = self._rand_pos()
                if (self.agent_start_pos != self.goal_pos):
                    break
        if direction is not None:
            self.agent_start_dir = direction
        else:
            self.agent_start_dir = random.randint(0, 3)

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(self.size, self.size)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.size, self.size)

        # Place a goal square in a random position
        self.place_goal()

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Generate N random blocks
        self.blocks = []
        for _ in range(self.num_tiles):
            # Ensure that the block doesn't overlap with the agent or goal
            while True:
                x, y = self._rand_pos()
                if ((x, y) != self.agent_start_pos) and ((x, y) != self.goal_pos) and ((x, y) not in self.blocks):
                    break
            self.blocks.append((x, y))
            self.grid.set(x, y, Wall())

        self.mission = "grand mission"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = None
        self.agent_dir = None
        self.agent_start_pos = None
        self.agent_start_dir = None

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.size, self.size)

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

        info = {'episode': {'r': reward, 'l': self.step_count, 't': terminated, 'truncated': truncated}}

        return obs, reward, terminated, truncated, info


def main():
    env = Env(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()