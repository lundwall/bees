import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CommunicationNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, comm_size=8, embedding_size=16, hidden_size=64):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.COMM_SIZE = comm_size
        self.EMBEDDING_SIZE = embedding_size
        self.HIDDEN_SIZE = hidden_size

        self.shared_mlp = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(137, 256), # 41 = 32 (embedding) + 9 (s_own)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 23)  # 7 possible actions + 16 new-state values (mean + log_std for each of the 8 comm values)
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(137, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value output
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict['obs'].shape[0]
        seq_len = 37  # excluding s_own

        # First vector in obs: s_own (8 values) | nectar_own (1 value)
        s_own = input_dict['obs'][:, 0, :9]  # shape: (batch_size, 9)
        sequences = input_dict['obs'][:, 1:, :]  # shape: (batch_size, 37, 16)

        # Join all batches together
        reshaped_sequences = sequences.reshape(-1, 16) # shape: (batch_size * 37, 16)

        # Pass each 16-value vector through the shared MLP
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


class SimpleAttention(nn.Module):
    def __init__(self, input_dim, query_dim):
        super(SimpleAttention, self).__init__()
        self.key_layer = nn.Linear(input_dim, query_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.query = nn.Parameter(torch.randn(query_dim))  # Learnable query

    def forward(self, x):
        keys = self.key_layer(x)
        values = self.value_layer(x)
        scores = torch.matmul(keys, self.query.unsqueeze(1).unsqueeze(0).transpose(-2, -1))
        attn_weights = torch.nn.functional.softmax(scores, dim=1)
        aggregated_vector = torch.sum(attn_weights * values, dim=1)
        return aggregated_vector


class CommunicationAttentionNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, comm_size=8, embedding_size=16, hidden_size=64):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=8, batch_first=True)

        self.action_mlp = nn.Sequential(
            nn.Linear(41, 64), # 41 = 32 (embedding) + 9 (s_own)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 23)  # 7 possible actions + 16 new-state values (mean + log_std for each of the 8 comm values)
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(41, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict['obs'].shape[0]
        seq_len = 37  # excluding s_own

        # First vector in obs: s_own (8 values) | nectar_own (1 value)
        s_own = input_dict['obs'][:, 0, :9]  # shape: (batch_size, 9)
        sequences = input_dict['obs'][:, 1:, :]  # shape: (batch_size, 37, 16)

        # Apply multi-head attention
        attn_output, _ = self.attention(sequences, sequences, sequences)

        # Reshape the output back to the original shape and sum along the sequence dimension
        # (one sum for each batch)
        summed_output = attn_output.view(batch_size, seq_len, -1).sum(dim=1)

        # Concatenate with own bee's state
        concatenated_output = torch.cat([summed_output, s_own], dim=1)  # shape: (batch_size, 25)

        # Pass the summed+concated output through the action MLP
        action_logits = self.action_mlp(concatenated_output)
        # Pass the summed+concated output through the value MLP
        self._value_out = self.value_mlp(concatenated_output)

        return action_logits, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])


class MultiHeadAttention(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.embedding_size = 16  # Size of each input vector
        self.num_heads = 8
        
        # MultiheadAttention module
        self.attention = nn.MultiheadAttention(self.embedding_size, self.num_heads)
        
        # Fully connected layer after attention
        self.fc = nn.Linear(self.embedding_size, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()  # shape: (batch_size, seq_len, embedding_size)
        
        # MultiheadAttention expects input shape: (seq_len, batch_size, embedding_size)
        x = x.permute(1, 0, 2)
        
        # queries, keys, and values are all the same
        attn_output, _ = self.attention(x, x, x)
        
        # Aggregate the output
        aggregated_output = attn_output.mean(dim=0)
        
        logits = self.fc(aggregated_output)
        return logits, state

    def value_function(self):
        pass


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_layer = nn.Linear(input_dim, input_dim * num_heads)
        self.value_layer = nn.Linear(input_dim, input_dim * num_heads)
        self.query = nn.Parameter(torch.randn(num_heads, input_dim))
        self.output_layer = nn.Linear(input_dim * num_heads, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        keys = self.key_layer(x).view(batch_size, seq_len, self.num_heads, -1)
        values = self.value_layer(x).view(batch_size, seq_len, self.num_heads, -1)
        
        outputs = []
        for i in range(self.num_heads):
            scores = torch.matmul(keys[:, :, i, :], self.query[i])
            attn_weights = torch.nn.functional.softmax(scores, dim=1)
            output = torch.sum(attn_weights.unsqueeze(-1) * values[:, :, i, :], dim=1)
            outputs.append(output)
        
        concatenated_outputs = torch.cat(outputs, dim=-1)
        return self.output_layer(concatenated_outputs)


class MultiHeadAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.embedding_size = 16
        self.num_heads = 8
        
        self.attention = CustomMultiHeadAttention(self.embedding_size, self.num_heads)
        self.fc = nn.Linear(self.embedding_size, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = self.attention(x)
        logits = self.fc(x)
        return logits, state

    def value_function(self):
        pass
