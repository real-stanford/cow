from typing import Any, Dict, List

import numpy as np
import torch
from allenact.base_abstractions.sensor import Sensor
from allenact_plugins.habitat_plugin.habitat_environment import \
    HabitatEnvironment
from allenact_plugins.habitat_plugin.habitat_tasks import ObjectNavTask
from src.simulation.constants import VOXEL_SIZE_M


class ExplorationTask(ObjectNavTask):
    def __init__(
        self,
        env: HabitatEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        rewards_config: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(env, sensors, task_info, max_steps, **kwargs)

        self.visited = set()

        assert "reward_type" in rewards_config
        self.reward_type = rewards_config["reward_type"]

        allowable_rewards = set(["supervised",])
        assert self.reward_type in allowable_rewards

        self.rewards_config = rewards_config
        self.prev_observation = self.get_observations()

    def judge(self) -> float:
        """Compute the reward after having taken a step. Overrides the Habitat ObjectNav Reward"""
        reward = self.rewards_config["step_penalty"]

        if self.reward_type == "supervised":
            reward += self._supervised_reward()
        else:
            raise ValueError("unsupported reward")

        self.prev_observation = self.get_observations()

        return float(reward)

    def shaping(self) -> float:
        return 0.0

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = {
            # "success": self._success,
            "total_reward": np.sum(self._rewards),
            "dist_to_target": 0,
            "spl": 0,
            "num_visited": len(self.visited),
        }
        return metrics

    def _supervised_reward(self):
        reward = 0.

        # never succeed as no success criteria for exploration
        self._success = False

        # get location and discretize to get a voxel coord
        curr_position = tuple(np.ndarray.round(
            self.env.get_location() / VOXEL_SIZE_M).tolist())

        if curr_position not in self.visited:
            reward += self.rewards_config["visited_reward"]
        self.visited.add(curr_position)

        if not self.last_action_success:
            reward += self.rewards_config["failed_action_penalty"]

        if self._took_end_action:
            reward += (
                self.rewards_config["goal_success_reward"]
                if self._success
                else self.rewards_config["failed_stop_reward"]
            )

        self._rewards.append(float(reward))

        return reward
