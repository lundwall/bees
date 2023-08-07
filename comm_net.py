import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CommunicationNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.shared_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 7 possible actions
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size, seq_len, _ = input_dict['obs'].shape
        reshaped_obs = input_dict['obs'].reshape(-1, 16)

        # Pass each 16-value vector through the shared MLP
        shared_output = self.shared_mlp(reshaped_obs)

        # Reshape the output back to the original shape and sum along the sequence dimension
        summed_output = shared_output.view(batch_size, seq_len, -1).sum(dim=1)

        # Pass the summed output through the action MLP
        action_logits = self.action_mlp(summed_output)
        # Pass the summed output through the value MLP
        self._value_out = self.value_mlp(summed_output)

        return action_logits, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])
