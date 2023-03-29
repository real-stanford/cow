# modified from: https://github.com/allenai/allenact/blob/main/projects/objectnav_baselines/experiments/objectnav_base.py
from abc import ABC
from typing import Optional, Sequence, Union

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.experiment_utils import Builder
import numpy as np


class BaseConfig(ExperimentConfig, ABC):
    """The base object navigation configuration file."""

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    VISIBILITY_DISTANCE = 1.0
    STOCHASTIC = True
    HORIZONTAL_FIELD_OF_VIEW = 90

    CAMERA_WIDTH = 224#672
    CAMERA_HEIGHT = 224#672
    SCREEN_SIZE = 224
    MAX_STEPS = 500

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Sequence[Sensor] = []

    def __init__(self):
        self.REWARDS_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
            "failed_action_penalty": -0.03,
            "visited_reward": 0.1,
            "object_reward": 0.4,
        }

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()
