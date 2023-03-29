from typing import Any, Dict, List

import numpy as np
import torch
from allenact.base_abstractions.sensor import Sensor
from allenact_plugins.robothor_plugin.robothor_environment import \
    RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from src.simulation.constants import VOXEL_SIZE_M


class ExplorationTask(ObjectNavTask):

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        rewards_config: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(env, sensors, task_info, max_steps, rewards_config, **kwargs)

        all_locations = [[k['x'], k['y'], k['z']]
                         for k in self._get_reachable_positions()]

        self.all_reachable_positions = torch.Tensor(all_locations)
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
        self.seen_objects = set()

        assert "reward_type" in rewards_config
        self.reward_type = rewards_config["reward_type"]
        self.num_objects = len(self.env.all_objects())

        allowable_rewards = set(["supervised", ])
        assert self.reward_type in allowable_rewards

        self.rewards_config = rewards_config
        self.prev_observation = self.get_observations()

        self.visited = set()

    def judge(self) -> float:
        """Compute the reward after having taken a step. Overrides the RoboTHOR ObjectNav Reward"""
        reward = self.rewards_config["step_penalty"]

        if self.reward_type == "supervised":
            reward += self._supervised_reward()
        else:
            raise ValueError("unsupported reward")

        self.prev_observation = self.get_observations()

        # add collision cost, maybe distance to goal objective,...
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
            "frac_map_visited": torch.sum(self.has_visited) / self.has_visited.shape[0],
            "frac_objects_seen": len(self.seen_objects) / self.num_objects,
            "num_visited": len(self.visited)
        }
        return metrics

    # def _supervised_reward(self):
    #     reward = 0.

    #     # never succeed as no success criteria for exploration
    #     self._success = False

    #     current_agent_location = self._get_agent_location()
    #     current_agent_location = torch.Tensor(
    #         [current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
    #     all_distances = self.all_reachable_positions - current_agent_location
    #     all_distances = (all_distances ** 2).sum(dim=-1)
    #     location_index = torch.argmin(all_distances)
    #     if self.has_visited[location_index] == 0:
    #         reward += self.rewards_config["visited_reward"]
    #     self.has_visited[location_index] = 1

    #     for o in self.env.visible_objects():
    #         self.seen_objects.add(o['name'])

    #     if not self.last_action_success:
    #         reward += self.rewards_config["failed_action_penalty"]

    #     if self._took_end_action:
    #         reward += (
    #             self.rewards_config["goal_success_reward"]
    #             if self._success
    #             else self.rewards_config["failed_stop_reward"]
    #         )

    #     self._rewards.append(float(reward))

    #     return reward

    def _supervised_reward(self):
        reward = 0.

        # never succeed as no success criteria for exploration
        self._success = False

        # get location and discretize to get a voxel coord
        current_agent_location = self._get_agent_location()
        current_agent_location = np.array(
            [current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])

        curr_position = tuple(np.ndarray.round(
            current_agent_location / VOXEL_SIZE_M).tolist())

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

    def _get_reachable_positions(self):
        event = self.env.controller.step('GetReachablePositions')
        # reachable_positions = event.metadata['reachablePositions']
        reachable_positions = event.metadata['actionReturn']

        # if reachable_positions is None or len(reachable_positions) == 0:
        #     reachable_positions = event.metadata['actionReturn']
        if reachable_positions is None or len(reachable_positions) == 0:
            print('Scene name',
                  self.env.controller.last_event.metadata['sceneName'])

        return reachable_positions

    def _get_agent_location(self):
        event = self.env.controller.step(action="Done")
        agent_position = event.metadata["agent"]["position"]

        return agent_position
