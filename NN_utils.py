import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import torch.nn as nn

import torch
import torch.nn as nn

# Define Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLPBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            Swish(),
            nn.Linear(dim_out, dim_out),
            nn.LayerNorm(dim_out),
            Swish(),
        )

    def forward(self, x):
        return self.block(x)

class GenericNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GenericNN, self).__init__()
        D = hidden_dims[0]
        assert all(h == D for h in hidden_dims), "All hidden dims must be the same for residuals"

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm_input = nn.LayerNorm(input_dim)
        self.input_layer = nn.Linear(input_dim, D)

        layers = []
        for _ in hidden_dims:
            layers.append(MLPBlock(D, D))
        self.res_blocks = nn.Sequential(*layers)

        self.output_layer = nn.Linear(D, output_dim)
        self.denorm_output = nn.Identity()

    def forward(self, x, time_emb,always_time=True):
        x = self.norm_input(x)
        x = self.input_layer(x)
        x = x + time_emb

        for block in self.res_blocks:
            if always_time:
                x = x + block(x + time_emb)
            else:
                x = x + block(x)

        x = self.output_layer(x)
        x = self.denorm_output(x)
        return x

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "model": self.res_blocks  # Fixed to reflect actual submodule
        }

#
# class GenericNN(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, activation='SiLU', final_activation=None):
#         super(GenericNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # Define activation functions
#         self.activation_fn = self.get_activation(activation)
#         self.final_activation_fn = self.get_activation(final_activation) if final_activation else None
#
#         # Input normalization
#         self.norm_input = nn.LayerNorm(input_dim)
#
#         # Create network layers
#         layers = []
#         last_dim = input_dim
#         for h_dim in hidden_dims:
#             layers.append(nn.Linear(last_dim, h_dim))
#             layers.append(self.activation_fn)
#             last_dim = h_dim
#         layers.append(nn.Linear(last_dim, output_dim))
#         if self.final_activation_fn:
#             layers.append(self.final_activation_fn)
#         self.layers = layers
#         self.model = nn.Sequential(*layers)
#
#         # Output denormalization — optional (identity by default)
#         self.denorm_output = nn.Identity()
#
#     def get_activation(self, name):
#         if name == 'relu':
#             return nn.ReLU()
#         elif name == 'SiLU':
#             return nn.SiLU()
#         elif name == 'tanh':
#             return nn.Tanh()
#         elif name == 'sigmoid':
#             return nn.Sigmoid()
#         elif name == 'leaky_relu':
#             return nn.LeakyReLU(0.2)
#         elif name == 'gelu':
#             return nn.GELU()
#         elif name is None:
#             return None
#         else:
#             raise ValueError(f"Unsupported activation: {name}")
#
#     def forward(self, x):
#         x = self.norm_input(x)
#         x = self.model(x)
#         x = self.denorm_output(x)
#         return x
#
#     def get_config(self):
#         return {
#             "input_dim": self.input_dim,
#             "output_dim": self.output_dim,
#             "model": self.model
#         }


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # For scalar timesteps, we need to start with a smaller dimension and project up
        self.dim = dim
        self.time_dim = 32  # Start with a smaller dimension for the time embedding

        # Projection layers
        self.proj1 = nn.Linear(self.time_dim, dim * 4)
        self.proj2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        # Ensure t is the right shape (batch_size, 1)
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)

        # batch_size = t.shape[0]

        # Create sinusoidal position embedding
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)

        # Shape: (batch_size, 1, half_dim)
        emb = t * emb.unsqueeze(0)

        # Apply sin and cos and reshape
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Now emb shape should be (batch_size, time_dim)

        # Project to higher dimension and back
        emb = self.proj1(emb)
        emb = F.silu(emb)
        emb = self.proj2(emb)

        return emb