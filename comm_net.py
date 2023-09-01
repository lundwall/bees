import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CommunicationNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.embedding_size = kwargs["embedding_size"]
        self.hidden_size = kwargs["hidden_size"]

        self.shared_mlp = nn.Sequential(
            nn.Linear(17, self.hidden_size), # 17 = 6 ohe + 8 comm + 3 nectar
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embedding_size),
            nn.ReLU()
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(self.embedding_size + 9, self.hidden_size), # 41 = 32 (embedding) + 9 (s_own)
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 7 + 16)  # 7 possible actions + 16 new-state values (mean + log_std for each of the 8 comm values)
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(self.embedding_size + 9, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)  # Single value output
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict['obs'].shape[0]
        seq_len = 37  # excluding s_own

        # First vector in obs: s_own (8 values) | nectar_own (1 value)
        s_own = input_dict['obs'][:, 0, :9]  # shape: (batch_size, 9)
        sequences = input_dict['obs'][:, 1:, :]  # shape: (batch_size, 37, 17)

        # Join all batches together
        reshaped_sequences = sequences.reshape(-1, 17) # shape: (batch_size * 37, 17)

        # Pass each 17-value vector through the shared MLP
        shared_output = self.shared_mlp(reshaped_sequences)

        # Reshape the output back to the original shape and sum along the sequence dimension
        # (one sum for each batch)
        summed_output = shared_output.view(batch_size, seq_len, -1).sum(dim=1)

        # Concatenate with own bee's state
        concatenated_output = torch.cat([summed_output, s_own], dim=1)  # shape: (batch_size, 25)

        # Pass the summed+concated output through the action MLP
        action_logits = self.action_mlp(concatenated_output)
        # Pass the summed+concated output through the value MLP
        self._value_out = self.value_mlp(concatenated_output)

        return action_logits, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_heads)])
        self.queries = nn.Parameter(torch.randn(num_heads, input_dim))
        self.output_layer = nn.Linear(input_dim * num_heads, input_dim)

    def forward(self, x):
        outputs = []
        for i in range(self.num_heads):
            keys = self.key_layers[i](x)
            values = self.value_layers[i](x)
            scores = torch.matmul(keys, self.queries[i])
            attn_weights = torch.nn.functional.softmax(scores, dim=1)
            aggregated_vector = torch.sum(attn_weights.unsqueeze(-1) * values, dim=1)
            outputs.append(aggregated_vector)
        concatenated_outputs = torch.cat(outputs, dim=-1)
        combined_output = self.output_layer(concatenated_outputs)
        return combined_output

class AttentionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, embedding_size=16, hidden_size=64, num_heads=8):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.attention = MultiHeadAttention(input_dim=16, num_heads=8)
        
        self.policy_net = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 7 + 16)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        
        # Identify non-padded sequences
        non_padded_mask = x.abs().sum(dim=-1) != 0  # Shape: [batch_size, seq_len]
        
        # Use the mask to filter out padded vectors
        non_padded_x = [seq[mask] for seq, mask in zip(x, non_padded_mask)]
        
        # Process each sequence through attention
        attn_outputs = []
        for seq in non_padded_x:
            attn_output = self.attention(seq)
            attn_outputs.append(attn_output)
        
        # Stack the attention outputs to form a single tensor
        stacked_attn_output = torch.stack(attn_outputs)
        
        logits = self.policy_net(stacked_attn_output)
        self._value_out = self.value_net(stacked_attn_output)
        return logits, state

    def value_function(self):
        return self._value_out.squeeze(1)


class SelfAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, embedding_size=16, hidden_size=64, num_heads=8):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.attention = MultiHeadAttention(input_dim=17, num_heads=8)
        
        self.policy_net = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        non_padded_mask = x.abs().sum(dim=-1) != 0
        # Keep only non-padded elements for each batch
        x = [data[mask] for data, mask in zip(x, non_padded_mask)]
        attn_output = self.attention(x)
        logits = self.policy_net(attn_output)
        self._value_out = self.value_net(attn_output)
        return logits, state

    def value_function(self):
        return self._value_out.squeeze(1)