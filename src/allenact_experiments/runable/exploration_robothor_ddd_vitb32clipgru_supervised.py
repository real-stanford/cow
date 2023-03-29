import torch.nn as nn
from src.allenact_experiments.shared.exploration_mixin_clipgru_config import \
    ExplorationMixInCLIPGRUConfig
from src.allenact_experiments.exploration.robothor.exploration_robothor_base_config import \
    ExplorationRoboThorBaseConfig
from src.allenact_experiments.shared.mixin_ddppo_config import MixInDDPPOConfig
from src.allenact_experiments.shared.vision_sensor import DepthSensorThor
from src.simulation.constants import CLIP_MEAN, CLIP_STD


class ExplorationRoboThorDDDViTB32Supervised(
    ExplorationRoboThorBaseConfig, MixInDDPPOConfig, ExplorationMixInCLIPGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        DepthSensorThor(
            height=ExplorationRoboThorBaseConfig.SCREEN_SIZE,
            width=ExplorationRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            # mean=CLIP_MEAN,
            # stdev=CLIP_STD,
            uuid="depth",
        ),
    ]

    def __init__(self):
        super().__init__()
        self.REWARDS_CONFIG["reward_type"] = "supervised"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return super().create_model(**dict(kwargs, clip_type="ViT-B/32"))

    @classmethod
    def tag(cls):
        return cls.__name__
