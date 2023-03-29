import platform
from datetime import datetime
from typing import Optional, Tuple

import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import (ActorCriticModel,
                                                      DistributionType,
                                                      LinearActorHead,
                                                      LinearCriticHead, Memory,
                                                      ObservationType)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.model_utils import compute_cnn_output
from gym.spaces.dict import Dict as SpaceDict

from src.clip import clip
from src.simulation.constants import CLIP_MODEL_TO_FEATURE_DIM


class ClipActorCriticDepth(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        obj_state_embedding_size=512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
        clip_type="RN50",
        clip_frozen=True,
        visualize=False,
    ):
        """Initializer.
        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.visualize = visualize

        # NOTE: for now assuming CLIP is frozen backbone as in EmbCLIP
        assert clip_frozen

        self.hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        model, _ = clip.load(clip_type)

        self.clip_frozen = clip_frozen

        if self.clip_frozen:
            for param in model.visual.parameters():
                param.requires_grad = False
            for param in model.transformer.parameters():
                param.requires_grad = False

        self.ddd_encoder = model.visual
        self.ddd_encoder.eval()

        self.state_encoder = RNNStateEncoder(
            CLIP_MODEL_TO_FEATURE_DIM[clip_type],
            self.hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_head = LinearActorHead(self.hidden_size, action_space.n)
        self.critic_head = LinearCriticHead(self.hidden_size)

        self.state_encoder.train()
        self.actor_head.train()
        self.critic_head.train()

        self.starting_time = datetime.now().strftime(
            "{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self.hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).
        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """
        # NOTE: this might not be a good idea lol
        ddd_observation = None
        if next(self.ddd_encoder.parameters()).is_cuda:
            ddd_observation = observations['depth'].repeat(1, 1, 1, 1, 3).half()
        else:
            ddd_observation = observations['depth'].repeat(1, 1, 1, 1, 3).float()


        ddd_observation_encoding = compute_cnn_output(
            self.ddd_encoder, ddd_observation)

        ddd_observation_encoding = ddd_observation_encoding / \
            ddd_observation_encoding.norm(dim=-1, keepdim=True)

        x_out, rnn_hidden_states = self.state_encoder(
            ddd_observation_encoding.float(), memory.tensor("rnn"), masks
        )

        actor_out = self.actor_head(x_out)
        critic_out = self.critic_head(x_out)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)

        return (
            actor_critic_output,
            memory,
        )
