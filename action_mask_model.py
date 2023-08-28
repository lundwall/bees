from gymnasium.spaces import Dict
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from comm_net import CommunicationNetwork, CommunicationAttentionNetwork
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
        comm_learning=False,
        comm_size=8,
        embedding_size=16,
        hidden_size=64,
        with_attn=False,
        **kwargs,
    ):
        self.comm_learning = comm_learning

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        if self.comm_learning:
            if with_attn:
                self.internal_model = CommunicationAttentionNetwork(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                    comm_size=comm_size,
                    embedding_size=embedding_size,
                    hidden_size=hidden_size,
                )
            else:
                self.internal_model = CommunicationNetwork(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                    comm_size=comm_size,
                    embedding_size=embedding_size,
                    hidden_size=hidden_size,
                )
        else:
                self.internal_model = ComplexInputNetwork(
                    orig_space["observations"],
                    action_space,
                    num_outputs,
                    model_config,
                    name + "_internal",
                )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

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
