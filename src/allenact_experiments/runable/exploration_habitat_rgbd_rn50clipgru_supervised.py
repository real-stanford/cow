import torch.nn as nn
from src.allenact_experiments.shared.exploration_mixin_clipgru_config import \
    ExplorationMixInCLIPGRUConfig
from src.allenact_experiments.exploration.habitat.exploration_habitat_base_config import \
    ExplorationHabitatBaseConfig
from src.allenact_experiments.shared.mixin_ddppo_config import MixInDDPPOConfig
from src.allenact_experiments.shared.vision_sensor import (DepthSensorHabitat,
                                                           RGBSensorHabitat)
from src.simulation.constants import CLIP_MEAN, CLIP_STD
from allenact_plugins.habitat_plugin.habitat_sensors import AgentCoordinatesSensorHabitat



class ExplorationHabitatRGBDRN50Supervised(
    ExplorationHabitatBaseConfig, MixInDDPPOConfig, ExplorationMixInCLIPGRUConfig,
):
    """An Object Navigation experiment configuration in Habitat with RGB
    input."""

    SENSORS = [
        RGBSensorHabitat(
            height=ExplorationHabitatBaseConfig.SCREEN_SIZE,
            width=ExplorationHabitatBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=CLIP_MEAN,
            stdev=CLIP_STD,
            uuid="rgb",
        ),
        DepthSensorHabitat(
            height=ExplorationHabitatBaseConfig.SCREEN_SIZE,
            width=ExplorationHabitatBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth",
        ),
        AgentCoordinatesSensorHabitat()
    ]

    def __init__(self):
        super().__init__()
        self.REWARDS_CONFIG["reward_type"] = "supervised"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return super().create_model(**dict(kwargs, clip_type="RN50"))

    @classmethod
    def tag(cls):
        return cls.__name__
