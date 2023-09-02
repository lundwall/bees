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
            scores = torch.matmul(keys, self.queries[i].unsqueeze(-1)).squeeze(-1)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            aggregated_vector = torch.sum(attn_weights.unsqueeze(1) * values, dim=0)
            outputs.append(aggregated_vector)
        concatenated_outputs = torch.cat(outputs, dim=-1)
        combined_output = self.output_layer(concatenated_outputs)
        return combined_output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.key_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_heads)])
        self.query_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_heads)])
        self.output_layer = nn.Linear(input_dim * num_heads, input_dim)

    def forward(self, x):
        outputs = []
        for i in range(self.num_heads):
            keys = self.key_layers[i](x)
            values = self.value_layers[i](x)
            queries = self.query_layers[i](x)
            scores = torch.matmul(queries, keys.transpose(-2, -1))
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            aggregated_vector = torch.matmul(attn_weights, values)
            outputs.append(aggregated_vector)
        concatenated_outputs = torch.cat(outputs, dim=-1)
        combined_output = self.output_layer(concatenated_outputs)
        return combined_output

class AttentionNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.embedding_size = kwargs["embedding_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.num_heads = kwargs["num_heads"]
        self.with_self_attn = kwargs["with_self_attn"]

        if self.with_self_attn:
            self.attention = MultiHeadSelfAttention(input_dim=17, num_heads=self.num_heads)
        else:
            self.attention = MultiHeadAttention(input_dim=17, num_heads=self.num_heads)
        
        self.policy_net = nn.Sequential(
            nn.Linear(17+9, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 7+16)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(17+9, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        
        special_vectors = x[:, 0, :9]  # Shape: [batch_size, 9]
        
        # Remove the special vectors from the input
        x = x[:, 1:, :]  # Shape: [batch_size, seq_len-1, input_dim]
        
        # Identify non-padded sequences
        non_padded_mask = x.abs().sum(dim=-1) != 0  # Shape: [batch_size, seq_len-1]
        
        # Handle the case where the entire tensor is zeros (e.g., during a dry run)
        if non_padded_mask.sum() == 0:
            self._value_out = torch.zeros((x.shape[0]))
            return torch.zeros((x.shape[0], 7+16)), state

        # Process each sequence through attention
        attn_outputs = []
        for seq, mask in zip(x, non_padded_mask):
            non_padded_seq = seq[mask]
            attn_output = self.attention(non_padded_seq)

            # Compute the average if in SA, since self-attention means that the
            # output is a sequence of as many vectors as the input
            if self.with_self_attn:
                attn_output = torch.mean(attn_output, dim=0)
            
            attn_outputs.append(attn_output)

        # Stack the attention outputs to form a single tensor
        stacked_attn_output = torch.stack(attn_outputs)
        
        # Concatenate the special vectors to the attention output
        combined_output = torch.cat([stacked_attn_output, special_vectors], dim=-1)

        logits = self.policy_net(combined_output)
        self._value_out = self.value_net(combined_output)

        return logits, state


    def value_function(self):
        return torch.reshape(self._value_out, [-1])
