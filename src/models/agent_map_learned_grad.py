import json
import os
from typing import List, Tuple

import torchvision.transforms as T
from PIL import Image
from src.allenact_experiments.runable.exploration_robothor_rgb_vitb32clipgru_supervised import \
    ExplorationRoboThorRGBViTB32Supervised
from src.allenact_experiments.shared.base_config import BaseConfig
from src.models.agent_build_utils import get_env_class_vars
from src.models.agent_map_learned import AgentMapLearned
from src.models.localization.clip_grad import ClipGrad
# from skimage import filters
from src.simulation.constants import (CLIP_MEAN, CLIP_STD, FORWARD_M, FOV,
                                      IMAGE_HEIGHT, IMAGE_WIDTH,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG,
                                      VOXEL_SIZE_M, IN_CSPACE)
from src.simulation.sim_enums import ClassTypes, EnvTypes
from src.simulation.utils import get_device
from torch import device

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AgentMapLearnedGrad(AgentMapLearned):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            clip_model_name: List[str],
            classes: List[str],
            classes_clip: List[str],
            templates: List[str],
            exploration_train_config: BaseConfig,
            exploration_model_path: str,
            fov: float,
            height: float,
            width: float,
            agent_height: float,
            floor_tolerance: float,
            threshold: float,
            device: device,
            max_ceiling_height: float = MAX_CEILING_HEIGHT_M,
            rotation_degrees: int = ROTATION_DEG,
            forward_distance: float = FORWARD_M,
            voxel_size_m: float = VOXEL_SIZE_M,
            in_cspace: bool = IN_CSPACE,
            debug_dir: str = None,
            wandb_log: bool = False,
            fail_stop: bool = True,
            open_clip_checkpoint: str = '',
            alpha: float = 0.,
            center_only: bool = False):

        super(AgentMapLearnedGrad, self).__init__(
            exploration_train_config=exploration_train_config,
            exploration_model_path=exploration_model_path,
            fov=fov,
            fail_stop=fail_stop,
            device=device,
            max_ceiling_height=max_ceiling_height,
            rotation_degrees=rotation_degrees,
            forward_distance=forward_distance,
            agent_height=agent_height,
            floor_tolerance=floor_tolerance,
            voxel_size_m=voxel_size_m,
            in_cspace=in_cspace,
            debug_dir=debug_dir,
            wandb_log=wandb_log,
            open_clip_checkpoint=open_clip_checkpoint,
            alpha=alpha)

        self.clip_module = ClipGrad(clip_model_name, classes,
                                  classes_clip, templates, threshold,
                                  device, center_only=center_only)

        if open_clip_checkpoint is not None and os.path.exists(open_clip_checkpoint):
            self.clip_module.load_weight_from_open_clip(open_clip_checkpoint, alpha)

        self.transform = T.Compose(
            [T.ToTensor(), T.Resize(224, interpolation=BICUBIC), T.Normalize(CLIP_MEAN, CLIP_STD)])


def build(exploration_model_path, prompts_path, fail_stop, threshold, clip_model_name="ViT-B/32", device_num=-1, debug_dir=None, wandb_log=False, env_type=EnvTypes.ROBOTHOR, class_type=ClassTypes.REGULAR, center_only=False):
    classes, classes_clip, agent_height, floor_tolerance, _, prompts = get_env_class_vars(prompts_path, env_type, class_type)

    agent_class = AgentMapLearnedGrad
    agent_kwargs = {
        "clip_model_name": clip_model_name,
        "classes": classes,
        "classes_clip": classes_clip,
        "templates": prompts,
        "exploration_train_config": ExplorationRoboThorRGBViTB32Supervised,
        "exploration_model_path": exploration_model_path,
        "fov": FOV,
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "agent_height": agent_height,
        "floor_tolerance": floor_tolerance,
        "device": get_device(device_num),
        "debug_dir": debug_dir,
        "wandb_log": wandb_log,
        "fail_stop": fail_stop,
        "threshold": threshold,
        "center_only": center_only,
    }

    render_depth = True

    return agent_class, agent_kwargs, render_depth
