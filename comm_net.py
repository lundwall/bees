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
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 23)  # 7 possible actions + 16 new-state values (mean + log_std for each of the 8 comm values)
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict['obs'].shape[0]
        seq_len = 37  # excluding the extra vector

        s_own = input_dict['obs'][:, 0, :9]  # shape: (batch_size, 9)
        sequences = input_dict['obs'][:, 1:, :]  # shape: (batch_size, 37, 16)

        # Join all batches together
        reshaped_sequences = sequences.reshape(-1, 16)

        # Pass each 16-value vector through the shared MLP
        shared_output = self.shared_mlp(reshaped_sequences)

        # Reshape the output back to the original shape and sum along the sequence dimension
        # (one sum for each batch)
        summed_output = shared_output.view(batch_size, seq_len, -1).sum(dim=1)

        # Concatenate with own bee's state
        concatenated_output = torch.cat([s_own, summed_output], dim=1)  # shape: (batch_size, 25)

        # Pass the summed+concated output through the action MLP
        action_logits = self.action_mlp(concatenated_output)
        # Pass the summed+concated output through the value MLP
        self._value_out = self.value_mlp(concatenated_output)

        return action_logits, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])
