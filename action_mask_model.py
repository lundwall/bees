from gymnasium.spaces import Dict
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from comm_net import CommunicationNetwork, AttentionModel, SelfAttentionModel
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        no_masking=False,
        comm_learning=False,
        with_attn=False,
        with_self_attn=False,
        embedding_size=16,
        hidden_size=64,
        num_heads=8,
    ):
        self.no_masking = no_masking
        self.comm_learning = comm_learning
        self.with_attn = with_attn
        self.with_self_attn = with_self_attn

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        if self.comm_learning:
            if self.with_self_attn:
                self.internal_model = SelfAttentionModel(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                    embedding_size=self.embedding_size,
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                )
            elif self.with_attn:
                self.internal_model = AttentionModel(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                    embedding_size=embedding_size,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                )
            else:
                self.internal_model = CommunicationNetwork(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                    embedding_size=self.embedding_size,
                    hidden_size=self.hidden_size,
                )
        else:
                self.internal_model = ComplexInputNetwork(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                )

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]["action_mask"].shape[0]
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        if self.comm_learning:
            # Here, two values for each: mean + log_std
            comm_mask = np.ones((batch_size, 16))
            action_mask = np.concatenate((action_mask, comm_mask), axis=1)
            action_mask = torch.from_numpy(action_mask).float()
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
