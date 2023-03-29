# modified from: https://github.com/allenai/allenact/blob/a26b3f074658f730f4de4cd946cd417114ae387b/projects/objectnav_baselines/experiments/objectnav_mixin_resnetgru.py

from typing import Sequence, Union

import gym
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from src.allenact_experiments.shared.base_config import BaseConfig
from src.allenact_experiments.shared.vision_sensor import (DepthSensor,
                                                           RGBSensor)
from src.models.exploration.simple_ac import SimpleActorCritic


class ExplorationMixInSimpleGRUConfig(BaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)

        assert has_rgb
        assert not has_depth

        return SimpleActorCritic(
            action_space=gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
        )
