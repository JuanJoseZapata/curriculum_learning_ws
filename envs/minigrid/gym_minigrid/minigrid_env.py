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
        width=10,
        height=10,
        num_tiles: int = 25,
        **kwargs,
    ):
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.width = width
        self.height = height
        self.num_tiles = num_tiles

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
        x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
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
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Place a goal square in a random position
        self.place_goal()

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Generate vertical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())

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
        self._gen_grid(self.width, self.height)

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


def main():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()