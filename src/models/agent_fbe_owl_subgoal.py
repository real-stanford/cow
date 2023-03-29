import json
import os
from typing import List, Tuple
from src.models.agent_build_utils import get_env_class_vars
from src.models.agent_mode import AgentMode

from src.models.localization.clip_owl import ClipOwl
from src.models.agent_fbe import AgentFbe
import torchvision.transforms as T
from PIL import Image
from src.simulation.constants import (FORWARD_M, FOV,
                                      IMAGE_HEIGHT, IMAGE_WIDTH,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG,
                                      VOXEL_SIZE_M, IN_CSPACE,
                                      THOR_PRIORS, LONGTAIL_PRIORS,
                                      HABITAT_PRIORS)
from src.simulation.sim_enums import ClassTypes, EnvTypes
from src.simulation.utils import get_device
from torch import device
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AgentFbeOwlSubgoal(AgentFbe):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            clip_model_name: List[str],
            classes: List[str],
            classes_clip: List[str],
            subgoals: List[str],
            templates: List[str],
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
            negate_action: bool = False,
            fail_stop: bool = True,
            open_clip_checkpoint: str = '',
            alpha: float = 0.,
            center_only: bool = False):

        super(AgentFbeOwlSubgoal, self).__init__(fov,
                                              device,
                                              max_ceiling_height=max_ceiling_height,
                                              rotation_degrees=rotation_degrees,
                                              forward_distance=forward_distance,
                                              agent_height=agent_height,
                                              floor_tolerance=floor_tolerance,
                                              voxel_size_m=voxel_size_m,
                                              in_cspace=in_cspace,
                                              debug_dir=debug_dir,
                                              wandb_log=wandb_log,
                                              negate_action=negate_action,
                                              fail_stop=fail_stop,
                                              open_clip_checkpoint=open_clip_checkpoint,
                                              alpha=alpha)

        self.clip_module = ClipOwl(clip_model_name, classes, subgoals,
                                       templates, threshold, device,
                                       center_only=center_only)

        if open_clip_checkpoint is not None and os.path.exists(open_clip_checkpoint):
            self.clip_module.load_weight_from_open_clip(
                open_clip_checkpoint, alpha)

        self.transform = T.Compose(
            [T.ToPILImage()])

        self.finding_subgoal = True
        self.classes = classes
        self.classes_clip = classes_clip

    def act(self, observations):
        action = super().act(observations)
        if action == 'Stop' and self.finding_subgoal:
            # found the subgoal
            self.finding_subgoal = False
            self.fbe.reset()
            self.agent_mode = AgentMode.SPIN
            self.last_action = None

            remap = {}
            for i in range(len(self.classes)):
                remap[self.classes[i]] = [self.classes_clip[i]]

            # map to final targets
            self.clip_module.remap_classes(remap)

            # re-act based on new mapping
            return super().act(observations)

        return action



def build(fail_stop, prompts_path, threshold, open_clip_checkpoint='', alpha=0., clip_model_name="ViT-B/32", device_num=-1, debug_dir=None, wandb_log=False, env_type=EnvTypes.ROBOTHOR, class_type=ClassTypes.REGULAR, center_only=False):

    classes, classes_clip, agent_height, floor_tolerance, negate_action, prompts = get_env_class_vars(prompts_path, env_type, class_type)

    subgoals = None
    if env_type == EnvTypes.ROBOTHOR or env_type == EnvTypes.NORMAL:
        subgoals = THOR_PRIORS
    elif env_type == EnvTypes.LONGTAIL:
        subgoals = LONGTAIL_PRIORS
    elif env_type == EnvTypes.HABITAT:
        subgoals = HABITAT_PRIORS
    else:
        assert False

    agent_class = AgentFbeOwlSubgoal
    agent_kwargs = {
        "clip_model_name": clip_model_name,
        "classes": classes,
        "classes_clip": classes_clip,
        "subgoals": subgoals,
        "templates": prompts,
        "fov": FOV,
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "agent_height": agent_height,
        "floor_tolerance": floor_tolerance,
        "device": get_device(device_num),
        "debug_dir": debug_dir,
        "wandb_log": wandb_log,
        "negate_action": negate_action,
        "fail_stop": fail_stop,
        "open_clip_checkpoint": open_clip_checkpoint,
        "alpha": alpha,
        "threshold": threshold,
        "center_only": center_only,
    }

    render_depth = True

    return agent_class, agent_kwargs, render_depth
