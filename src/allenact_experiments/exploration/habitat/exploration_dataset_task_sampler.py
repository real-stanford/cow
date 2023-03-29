import random
from typing import Any, Dict, List, Optional

import gym
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.system import get_logger
from allenact_plugins.habitat_plugin.habitat_environment import \
    HabitatEnvironment
from allenact_plugins.habitat_plugin.habitat_task_samplers import \
    ObjectNavTaskSampler
from src.allenact_experiments.exploration.habitat.exploration_task import \
    ExplorationTask
from allenact.utils.experiment_utils import set_deterministic_cudnn
from habitat.config import Config


class ExplorationDatasetTaskSampler(ObjectNavTaskSampler):
    def __init__(
        self,
        env_config: Config,
        sensors: List[Sensor],
        max_steps: int,
        action_space: gym.Space,
        distance_to_goal: float,
        rewards_config: Dict,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        loop_dataset: bool = True,
        allow_flipping=False,
        **kwargs,
    ) -> None:
        super().__init__(
            env_config,
            sensors,
            max_steps,
            action_space,
            distance_to_goal,
            **kwargs,
        )
        self.rewards_config = rewards_config
        self.seed = seed

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self._last_sampled_task: Optional[ExplorationTask] = None

    @property
    def last_sampled_task(self) -> Optional[ExplorationTask]:
        return self._last_sampled_task

    def next_task(self, force_advance_scene=False) -> Optional[ExplorationTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.env is not None:
            self.env.reset()
        else:
            self.env = self._create_environment()
            self.env.reset()
            self.set_seed(self.seed)
        ep_info = self.env.get_current_episode()
        target = ep_info.goals[0].position

        task_info = {
            "target": target,
            "distance_to_goal": self.distance_to_goal,
        }

        self._last_sampled_task = ExplorationTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            rewards_config=self.rewards_config,
        )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self._last_sampled_task