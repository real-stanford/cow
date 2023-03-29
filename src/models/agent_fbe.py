import os
from typing import List, Tuple
from abc import ABC, abstractmethod

# from skimage import filters
from src.models.agent import Agent
from src.models.exploration.frontier_based_exploration import FrontierBasedExploration
from src.simulation.constants import (FORWARD_M,
                                      MAX_CEILING_HEIGHT_M,
                                      ROTATION_DEG, VOXEL_SIZE_M, IN_CSPACE)
from torch import device, is_tensor
from src.models.agent_mode import AgentMode
from threadpoolctl import threadpool_limits
import numpy as np


class AgentFbe(Agent, ABC):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            fov: float,
            device: device,
            agent_height: float,
            floor_tolerance: float,
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
            alpha: float = 0.):

        super(AgentFbe, self).__init__()

        self.fbe = FrontierBasedExploration(fov, device, max_ceiling_height, agent_height,
                                            floor_tolerance, rotation_degrees, forward_distance,
                                            voxel_size_m, in_cspace, wandb_log, negate_action, fail_stop)
        self.timesteps = 0
        self.debug_dir = debug_dir
        self.debug_data = []
        if debug_dir is not None:
            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)

        self.agent_mode = AgentMode.SPIN

        self.action_queue = []

        self.rotation_degrees = rotation_degrees
        self.forward_distance = forward_distance
        assert (360-int(fov)) % self.rotation_degrees == 0
        self.rotation_counter = 0
        self.max_rotation_count = (360-int(fov))/self.rotation_degrees
        self.last_action = None
        self.open_clip_checkpoint = open_clip_checkpoint
        self.alpha = alpha

    def reset(self):
        self.timesteps = 0
        self.fbe.reset()
        self.agent_mode = AgentMode.SPIN
        self.last_action = None

    def act(self, observations):
        # analyse observation for object
        with threadpool_limits(limits=1):
            attention = self.localize_object(observations)

            self.debug_data.append((self.timesteps, attention.max().item()))

            # with threadpool_limits(limits=1):
            # update map
            self.fbe.update_map(
                observations,
                attention,
                self.last_action)

            if self.fbe.poll_roi_exists() and self.agent_mode != AgentMode.EXPLOIT:
                self.rotation_counter = 0
                self.action_queue = []

                # NOTE: uncomment for fig
                self.agent_mode = AgentMode.EXPLOIT

            elif self.agent_mode == AgentMode.SPIN and self.rotation_counter == self.max_rotation_count:
                self.rotation_counter = 0
                self.action_queue = []
                self.agent_mode = AgentMode.EXPLORE

            # determine action to take
            action = None

            if self.agent_mode == AgentMode.SPIN:
                action = self.rotate()
            elif self.agent_mode == AgentMode.EXPLORE:
                action = self.explore(observations, attention)
            elif self.agent_mode == AgentMode.EXPLOIT:
                action = self.exploit(observations, attention)

            self.timesteps += 1
            self.last_action = action

        return action

    def localize_object(self, observations) -> Tuple[int, float]:
        img_tensor = None

        if self.transform is not None:
            img_tensor = self.transform(observations["rgb"])
        else:
            img_tensor = observations["rgb"]

        # will always be true but not for ViT-OWL localization
        if is_tensor(img_tensor):
            img_tensor = img_tensor.unsqueeze(0)

        # NOTE: child must set clip_module
        return self.clip_module(img_tensor, observations["object_goal"])

    def rotate(self) -> str:
        self.rotation_counter += 1
        return "RotateLeft"

    def explore(self, observations, attention) -> str:
        if not len(self.action_queue):
            self.action_queue = self.fbe.actions_toward_next_frontier()

        if len(self.action_queue) == 0:
            # Agent confused, move to spin mode
            self.fbe.reset()
            self.agent_mode = AgentMode.SPIN
            self.fbe.update_map(
                observations,
                attention,
                self.last_action)

            return self.rotate()

        return self.action_queue.pop(0)

    def exploit(self, observations, attention) -> str:

        if not len(self.action_queue):
            self.action_queue = self.fbe.action_towards_next_roi()

        if len(self.action_queue) == 0:
            # Agent confused, move to spin mode
            self.fbe.reset()
            self.agent_mode = AgentMode.SPIN
            self.fbe.update_map(
                observations,
                attention,
                self.last_action)

            return self.rotate()

        return self.action_queue.pop(0)
