import os
from enum import IntEnum
from typing import List, Tuple
from abc import ABC, abstractmethod
from src.allenact_experiments.shared.base_config import BaseConfig

# from skimage import filters
from src.models.agent import Agent
from src.models.agent_mode import AgentMode
from src.simulation.constants import (FORWARD_M,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG, VOXEL_SIZE_M)
from torch import device
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
import torch
from torch.distributions.categorical import Categorical
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.algorithms.onpolicy_sync.policy import Memory
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from typing import Tuple, cast


class AgentLearned(Agent, ABC):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            exploration_train_config: BaseConfig,
            exploration_model_path: str,
            fov: float,
            device: device,
            rotation_degrees: int = ROTATION_DEG,
            forward_distance: float = FORWARD_M,
            debug_dir: str = None,
            wandb_log: bool = False):

        super(AgentLearned, self).__init__()

        self.agent_mode = AgentMode.EXPLORE

        self.exploration_agent = exploration_train_config.create_model()
        self.exploration_agent.load_state_dict(
            torch.load(exploration_model_path)['model_state_dict'])
        self.exploration_agent.eval()

        self.timesteps = 0
        self.debug_dir = debug_dir
        self.debug_data = []
        if debug_dir is not None:
            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)

        self.rotation_degrees = rotation_degrees
        self.forward_distance = forward_distance
        assert (360-int(fov)) % self.rotation_degrees == 0
        self.last_action = None
        self.failed_action = None
        self.last_observation = None
        self.device = device
        self.action_order = ObjectNavTask.class_action_names()

        self._reset_memory_and_mask(self.exploration_agent, self.device)

    def reset(self):
        self.timesteps = 0
        self.last_action = None
        self.failed_action = None
        self.last_observation = None
        self._reset_memory_and_mask(self.exploration_agent, self.device)

    def act(self, observations):

        new_obs = torch.as_tensor(observations["rgb"])

        if self.last_observation is not None:
            abs_diff = torch.abs(self.last_observation-new_obs)
            m_pix = torch.mean(abs_diff)
            s_pix = torch.std(abs_diff)
            if m_pix.item() < 0.1 and s_pix.item() < 0.1:
                self.failed_action = self.last_action
            else:
                self.failed_action = None

        self.last_observation = new_obs

        # analyse observation for object
        attention = self.localize_object(observations)

        index = torch.argmax(attention)

        self.debug_data.append((self.timesteps, attention[index].item()))

        # NOTE: attention is binary
        if attention[index].item() > 0.9:
            self.agent_mode = AgentMode.EXPLOIT

        # determine action to take
        action = None

        if self.agent_mode == AgentMode.EXPLORE:
            action = self.explore(observations, attention)
        elif self.agent_mode == AgentMode.EXPLOIT:
            action = self.exploit(observations, attention)
        else:
            raise ValueError('unsupported mode')

        self.timesteps += 1
        self.last_action = action

        return action

    def exploit(self, observations, attention) -> str:
        # NOTE: this gets overriden by child
        return "MoveAhead"

    def explore(self, observations, attention) -> str:

        img_tensor = self.transform(observations["rgb"]).permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(self.device)
        ob = {'rgb': img_tensor}

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

        dist = ac_out.distributions.probs
        if self.fbe.failed_action is not None and self.fail_stop:
            dist[:, :, self.action_order_inverse[self.fbe.failed_action]] = 0
        dist = Categorical(ac_out.distributions.probs[:, :, :3])

        action_num = dist.sample().item()
        action = self.action_order[action_num]

        return action

    def localize_object(self, observations) -> Tuple[int, float]:
        img_tensor = self.transform(observations["rgb"]).unsqueeze(0)

        # NOTE: child must set clip_module
        return self.clip_module(img_tensor, observations["object_goal"])

    def action_fail_check(self):
        pass

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
