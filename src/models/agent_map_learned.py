import os
from typing import List, Tuple
from abc import ABC

from habitat import SensorSuite
from src.allenact_experiments.shared.base_config import BaseConfig
from src.clip.clip import BICUBIC

# from skimage import filters
from src.models.agent_fbe import AgentFbe, AgentMode
from src.simulation.constants import (CLIP_MEAN, CLIP_STD, FORWARD_M,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG, VOXEL_SIZE_M, IN_CSPACE)
from torch import device
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
import torch
from torch.distributions.categorical import Categorical
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.algorithms.onpolicy_sync.policy import Memory
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from typing import Tuple, cast
from src.models.agent_mode import AgentMode
import torchvision.transforms as T


class AgentMapLearned(AgentFbe, ABC):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            exploration_train_config: BaseConfig,
            exploration_model_path: str,
            fov: float,
            agent_height: float,
            floor_tolerance: float,
            fail_stop: bool,
            device: device,
            max_ceiling_height: float = MAX_CEILING_HEIGHT_M,
            rotation_degrees: int = ROTATION_DEG,
            forward_distance: float = FORWARD_M,
            voxel_size_m: float = VOXEL_SIZE_M,
            in_cspace: bool = IN_CSPACE,
            debug_dir: str = None,
            wandb_log: bool = False,
            open_clip_checkpoint: str = '',
            alpha: float = 0.):

        super(AgentMapLearned, self).__init__(
            fov=fov,
            device=device,
            max_ceiling_height=max_ceiling_height,
            rotation_degrees=rotation_degrees,
            forward_distance=forward_distance,
            fail_stop=fail_stop,
            agent_height=agent_height,
            floor_tolerance=floor_tolerance,
            voxel_size_m=voxel_size_m,
            in_cspace=in_cspace,
            debug_dir=debug_dir,
            wandb_log=wandb_log,
            open_clip_checkpoint=open_clip_checkpoint,
            alpha=alpha)

        sensor_preprocessor_graph = SensorSuite(exploration_train_config.SENSORS)
        self.exploration_agent = exploration_train_config.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph, device=device)
        self.exploration_agent.load_state_dict(
            torch.load(exploration_model_path)['model_state_dict'])
        self.exploration_agent.eval()
        self.exploration_agent.to(device)

        self.timesteps = 0
        self.debug_dir = debug_dir
        self.debug_data = []
        if debug_dir is not None:
            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)

        self.device = device
        self.action_order = ObjectNavTask.class_action_names()
        self.action_order_inverse = {a: i for i, a in enumerate(self.action_order)}
        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
        self.transform_depth = T.ToTensor()

        self._reset_memory_and_mask(self.exploration_agent, self.device)
        self.fail_stop = fail_stop

    def reset(self):
        super().reset()
        self.timesteps = 0
        self.last_action = None
        self._reset_memory_and_mask(self.exploration_agent, self.device)

    def act(self, observations):

        # Should never be in spin mode as this is heuristic
        if self.agent_mode == AgentMode.SPIN:
            self.agent_mode = AgentMode.EXPLORE

        # analyse observation for object
        # observation['image'] = observation['image'].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(0)
        return super().act(observations)

    def explore(self, observations, attention) -> str:

        t = T.Compose(
            [T.ToTensor(), T.Resize(224, interpolation=BICUBIC), T.Normalize(CLIP_MEAN, CLIP_STD)])

        img_tensor = t(observations["rgb"]).permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(self.device)
        depth_tensor = self.transform_depth(observations["depth"]).permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(self.device)

        # NOTE: to be consistent with training
        depth_tensor = torch.clamp(depth_tensor, 0.0, 5.0) / 5.0 * 0.25

        ob = {'rgb': img_tensor, 'depth': depth_tensor}

        ac_out, self.memory = cast(
            Tuple[ActorCriticOutput, Memory],
            self.exploration_agent.forward(
                observations=ob,
                memory=self.memory,
                prev_actions=None,
                masks=self.masks,
            ),
        )

        self.masks.fill_(1)

        # dist = Categorical(ac_out.distributions.probs)
        dist = ac_out.distributions.probs
        if self.fbe.failed_action is not None and self.fail_stop:
            dist[:, :, self.action_order_inverse[self.fbe.failed_action]] = 0
        dist = Categorical(ac_out.distributions.probs[:, :, :3])

        action_num = dist.sample().item()
        action = self.action_order[action_num]

        return action


    def _reset_memory_and_mask(self, model, device):
        self.rollout_storage = RolloutStorage(
            num_steps=1,
            num_samplers=1,
            actor_critic=model,
            only_store_first_and_last_in_memory=True,
        )
        self.memory = self.rollout_storage.pick_memory_step(0)
        tmp = self.memory["rnn"][1]
        self.memory["rnn"] = (self.memory["rnn"][0].to(device), tmp)

        self.memory.tensor("rnn").to(device)
        self.masks = self.rollout_storage.masks[:1]
        self.masks = 0 * self.masks
        self.masks = self.masks.to(device)
