import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from NN_utils import GenericNN,TimeEmbedding


def kerras_boundaries(sigma_min, sigma_max, N, rho):
    steps = torch.arange(N + 1, dtype=torch.float32)
    return (sigma_max ** (-1 / rho) + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)


class ConsistencyModel(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int = 0, eps: float = 0.002, D: int = 128, depth: int = 6):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.D = D
        if cond_dim>0:
            self.cond_embed = nn.Linear(cond_dim, D)

        # Positional encoding frequency components
        # self.freqs = torch.exp(
        #     -math.log(10000) * torch.arange(start=0, end=D, dtype=torch.float32) / D
        # )

        # Optional conditioning
        self.input_layer = nn.Linear(input_dim , D)
        self.time_embed = TimeEmbedding(D)

        # Time embedding MLP
        # self.time_layers = nn.Sequential(
        #     nn.Linear(2 * D, D),
        #     nn.SiLU(),
        #     nn.Linear(D, D),
        # )

        # Residual network using GenericNN
        self.net = GenericNN(
            input_dim=D,
            hidden_dims=[D] * depth,  # depth controls number of residual layers
            output_dim=input_dim,
        )

    def forward(self, x, t, cond=None):
        B, C = x.shape
        x_ori = x
        x = self.input_layer(x)
        # Add conditioning if present
        if self.cond_dim>0 and cond is not None:
            cond_emb = self.cond_embed(cond)


            x = x + cond_emb


        # if cond is not None:
        #     x = torch.cat([x, cond], dim=1)

        # Ensure t is (B, 1)
        if isinstance(t, (float, int)):
            t = torch.tensor([t] * B, dtype=torch.float32, device=x.device).unsqueeze(1)

        time_emb = self.time_embed(t)
        x = self.net(x, time_emb)

        # Output skip connection with consistency weights
        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).sqrt()

        return c_skip_t * x_ori + c_out_t * x


    def loss(self, x, z, t0, t1, ema_model, diffusion=None, cond=None):
        x2 = x + z * t1
        x2 = self(x2, t1, cond=cond)

        with torch.no_grad():
            if diffusion is not None:
                x1 = diffusion.ode_step(x, t0, t1)
            else:
                x1 = x + z * t0
            x1 = ema_model(x1, t0, cond=cond)

        return F.mse_loss(x1, x2)

    @torch.no_grad()
    def sample(self, x, ts: List[float], cond=None):
        x = self(x, ts[0], cond=cond)

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=cond)

        return x

    def train_model(self, X: torch.Tensor = None, n_epoch: int = 100, device: str = "cpu", condition=None,
              model_diff=None):
        B, C = X.shape
        self.to(device)

        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        ema_model = ConsistencyModel(input_dim=self.input_dim, cond_dim=self.cond_dim, D=self.D)
        ema_model.to(device)
        ema_model.load_state_dict(self.state_dict())

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        pbar = tqdm(range(1, n_epoch + 1))
        for epoch in pbar:
            N = math.ceil(math.sqrt((epoch * (150 ** 2 - 4) / n_epoch) + 4) - 1) + 1
            boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

            loss_ema = None

            for (x,) in dataloader:
                x = x.to(device)

                if condition is not None and condition > 0:
                    assert condition < C, "Condition dimension must be less than input dimension"
                    cond_x = x[:, :condition]
                    x = x[:, condition:]
                else:
                    cond_x = None

                z = torch.randn_like(x)
                t_idx = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
                t0 = boundaries[t_idx]#.squeeze(1)
                t1 = boundaries[t_idx + 1]#.squeeze(1)

                optim.zero_grad()
                loss = self.loss(x, z, t0, t1, ema_model=ema_model, diffusion=model_diff, cond=cond_x)
                loss.backward()
                optim.step()

                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

                with torch.no_grad():
                    mu = math.exp(2 * math.log(0.95) / N)
                    for p, ema_p in zip(self.parameters(), ema_model.parameters()):
                        ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.6f}, mu: {mu:.4f}")
        torch.save(self.state_dict(), f"./ct_{n_epoch}_{condition}.pth")
        return self


# def train_consistency(
#         model: ConsistencyModel=None,
#         X: torch.Tensor=None,
#         n_epoch: int = 100,
#         device: str = "cpu",
#         name: str = "mog",
#         model_diff=None,
#         condition: int = 0,
# ):
#     def kerras_boundaries(sigma_min, sigma_max, N, rho):
#         steps = torch.arange(N + 1, dtype=torch.float32)
#         return (sigma_max ** (-1 / rho) + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)
#
#     B, C = X.shape
#
#     model.to(device)
#     optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
#
#     ema_model = ConsistencyModel(input_dim=model.input_dim, cond_dim=model.cond_dim, D=model.D)
#     ema_model.to(device)
#     ema_model.load_state_dict(model.state_dict())
#
#     dataset = TensorDataset(X)
#     dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
#     pbar = tqdm(range(1, n_epoch + 1))
#     for epoch in pbar:
#         N = math.ceil(math.sqrt((epoch * (150 ** 2 - 4) / n_epoch) + 4) - 1) + 1
#         boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)
#
#         loss_ema = None
#         model.train()
#
#         for (x,) in dataloader:
#             x = x.to(device)
#
#             cond_x = x[:, :condition] if condition > 0 else None
#             x = x[:, condition:] if condition < C else x
#
#             z = torch.randn_like(x)
#             t_idx = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
#             t0 = boundaries[t_idx]
#             t1 = boundaries[t_idx + 1]
#
#             optim.zero_grad()
#             loss = model.loss(x, z, t0, t1, ema_model=ema_model, diffusion=model_diff, cond=cond_x)
#             loss.backward()
#             optim.step()
#
#             if loss_ema is None:
#                 loss_ema = loss.item()
#             else:
#                 loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
#
#             with torch.no_grad():
#                 mu = math.exp(2 * math.log(0.95) / N)
#                 for p, ema_p in zip(model.parameters(), ema_model.parameters()):
#                     ema_p.mul_(mu).add_(p, alpha=1 - mu)
#
#         pbar.set_description(f"loss: {loss_ema:.6f}, mu: {mu:.4f}")
#     model.eval()
#     torch.save(model.state_dict(), f"./ct_{name}.pth")
#     return model
