import random
from typing import Any, Dict, List, Optional

import gym
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.system import get_logger
from allenact_plugins.robothor_plugin.robothor_environment import \
    RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_task_samplers import \
    ObjectNavDatasetTaskSampler
from src.allenact_experiments.exploration.robothor.exploration_task import \
    ExplorationTask


class ExplorationDatasetTaskSampler(ObjectNavDatasetTaskSampler):
    def __init__(
        self,
        scenes: List[str],
        scene_directory: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        loop_dataset: bool = True,
        allow_flipping=False,
        env_class=RoboThorEnvironment,
        **kwargs,
    ) -> None:
        super().__init__(
            scenes,
            scene_directory,
            sensors,
            max_steps,
            env_args,
            action_space,
            rewards_config,
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
            loop_dataset=loop_dataset,
            allow_flipping=allow_flipping,
            env_class=env_class,
            **kwargs,
        )

        self._last_sampled_task: Optional[ExplorationTask] = None

    @property
    def last_sampled_task(self) -> Optional[ExplorationTask]:
        return self._last_sampled_task

    def next_task(self, force_advance_scene: bool = False) -> Optional[ExplorationTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0
        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is None:
            self.env = self._create_environment()

        if scene.replace("_physics", "") != self.env.scene_name.replace("_physics", ""):
            self.env.reset(scene_name=scene)
        else:
            self.env.reset_object_filter()

        self.env.set_object_filter(
            object_ids=[
                o["objectId"]
                for o in self.env.last_event.metadata["objects"]
                if o["objectType"] == episode["object_type"]
            ]
        )

        task_info = {"scene": scene, "object_type": episode["object_type"]}
        if len(task_info) == 0:
            get_logger().warning(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(
                    scene, self.object_types)
            )
        task_info["initial_position"] = episode["initial_position"]
        task_info["initial_orientation"] = episode["initial_orientation"]
        task_info["initial_horizon"] = episode.get("initial_horizon", 0)
        task_info["distance_to_target"] = episode.get("shortest_path_length")
        task_info["path_to_target"] = episode.get("shortest_path")
        task_info["object_type"] = episode["object_type"]
        task_info["id"] = episode["id"]
        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1
        if not self.env.teleport(
            pose=episode["initial_position"],
            rotation=episode["initial_orientation"],
            horizon=episode.get("initial_horizon", 0),
        ):
            return self.next_task()
        self._last_sampled_task = ExplorationTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            rewards_config=self.rewards_config,
        )
        return self._last_sampled_task
