import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from NN_utils import GenericNN, TimeEmbedding


def kerras_boundaries(sigma_min, sigma_max, N, rho):
    steps = torch.arange(N + 1, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    return (sigma_max ** (-1 / rho) + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)


class ConsistencyModel(nn.Module):
    def __init__(self, nfeatures: int, condition_on: int = 0, eps: float = 0.002, nunits: int = 128, depth: int = 6,
                 device=None):
        super().__init__()
        self.eps = eps
        self.nfeatures = nfeatures
        self.condition_on = condition_on
        self.nunits = nunits
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # #### Changed - from Improved Consistency Training paper ###
        # Added sigma_data parameter for better parameterization
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.rho = 7.0
        ### End of changed ####

        self.betas = None

        if condition_on > 0:
            self.cond_embed = nn.Linear(condition_on, nunits)

        self.input_layer = nn.Linear(nfeatures, nunits)
        self.time_embed = TimeEmbedding(nunits)
        self.out = nn.Linear(nunits, nfeatures)
        self.net = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] * depth,
            output_dim=nunits,
        )

        self.to(self.device)

    def forward(self, x, t, cond=None):
        x_ori = x

        # #### Changed - from Improved Consistency Training paper ###
        # Better parameterization with improved skip connections
        # Old version:
        # x = self.input_layer(x)
        # ...
        # c_skip_t = 0.25 / (t.pow(2) + 0.25)
        # c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).sqrt()

        # New improved parameterization:
        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)

        c_skip = self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)
        c_out = self.sigma_data * t / torch.sqrt(t ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(t ** 2 + self.sigma_data ** 2)

        x_scaled = c_in * x_ori
        x = self.input_layer(x_scaled)
        ### End of changed ####

        # Add conditioning if present
        if self.condition_on > 0 and cond is not None:
            cond_emb = self.cond_embed(cond)
            x = x + cond_emb

        time_emb = self.time_embed(t)
        x = self.net(x, time_emb)
        x = self.out(x)

        # #### Changed - from Improved Consistency Training paper ###
        # New skip connection using improved parameterization
        return c_skip * x_ori + c_out * x
        ### End of changed ####

    # #### Changed - from Consistency Models paper (Consistency Training) ###
    # Old distillation loss that required teacher model:
    # def loss(self, x, z, t0, t1, ema_model, diffusion=None, cond=None,device="cpu"):
    #     with torch.no_grad():
    #         if diffusion is not None:
    #             x2,_= diffusion.noise(x, t1)
    #             x1,_ = diffusion.sample_ddim_step( x2, t1, condition_x=cond, device=device, eta=0.0)
    #         else:
    #             x2 = x + z * t1
    #             x1 = x + z * t0
    #         x1 = ema_model(x1, t0, cond=cond)
    #     x2 = self(x2, t1, cond=cond)
    #     return F.mse_loss(x1, x2)

    # New Consistency Training loss (no teacher model needed):
    def loss_ct(self, x, z, t0, t1, cond=None):
        """Direct Consistency Training loss"""
        x_t1 = x + z * t1
        x_t0 = x + z * t0

        f_theta_t1 = self(x_t1, t1, cond=cond)
        f_theta_t0 = self(x_t0, t0, cond=cond)

        return F.mse_loss(f_theta_t1, f_theta_t0)

    def multistep_loss(self, x, z, boundaries, cond=None):
        """Multistep consistency training from later papers"""
        losses = []
        N = len(boundaries) - 1

        # Sample multiple timestep pairs
        for _ in range(min(3, N - 1)):  # Use up to 3 pairs per batch
            i = torch.randint(0, N - 2, (1,)).item()
            t0 = boundaries[i]
            t1 = boundaries[i + 1]
            t2 = boundaries[i + 2] if i + 2 < len(boundaries) else boundaries[i + 1]

            loss1 = self.loss_ct(x, z, t0, t1, cond)
            if t2 != t1:
                loss2 = self.loss_ct(x, z, t0, t2, cond)
                losses.extend([loss1, loss2])
            else:
                losses.append(loss1)

        return torch.mean(torch.stack(losses)) if losses else self.loss_ct(x, z, boundaries[0], boundaries[1], cond)

    ### End of changed ####

    def adaptive_boundaries(self, epoch, max_epochs):
        """Adaptive curriculum learning for time boundaries"""
        max_N = 150
        min_N = 4
        progress = epoch / max_epochs
        N = int(min_N + (max_N - min_N) * progress ** 0.5)
        return kerras_boundaries(self.sigma_min, self.sigma_max, N, self.rho).to(self.device)

    def sample(self, nsamples=250, condition_x=None, ts: List[float] = [150.0, 50.0, 20.0, 10.0, 5.0, 1.], device=None):
        device = self.device if device is None else device
        if condition_x is not None:
            condition_x = condition_x.to(device)

        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        x = x.to(device)

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)

        return x, None, None

    def train_model(self, X: torch.Tensor = None, data_generator=None, nepochs: int = 100, batch_size: int = 2048,
                    device: str = "cpu", condition=None, model_diff=None, use_multistep=True):
        assert X is not None or data_generator is not None, "Need either X or data_generator"

        self.to(device)
        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        # #### Changed - from Consistency Training paper ###
        # Remove EMA model requirement for CT (only needed for CD)
        # Old version used ema_model for distillation
        # New version uses direct consistency training
        ### End of changed ####

        if X is not None:
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            use_dataloader = True
        else:
            use_dataloader = False

        pbar = tqdm(range(1, nepochs + 1))
        for epoch in pbar:
            # #### Changed - Adaptive curriculum learning ###
            # Old: Fixed schedule
            # N = math.ceil(math.sqrt((epoch * (150 ** 2 - 4) / nepochs) + 4) - 1) + 1
            # boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

            # New: Adaptive boundaries with curriculum learning
            boundaries = self.adaptive_boundaries(epoch, nepochs)
            ### End of changed ####

            self.betas = boundaries
            loss_ema = None

            if use_dataloader:
                data_iter = dataloader
            else:
                data_iter = [None]  # Single iteration per epoch

            for batch_idx in data_iter:
                if use_dataloader:
                    (x,) = batch_idx
                else:
                    x = data_generator(batch_size, device=device)

                x = x.to(device)

                if condition is not None and condition > 0:
                    assert condition < x.shape[1], "Condition dimension must be less than input dimension"
                    cond_x = x[:, :condition]
                    x = x[:, condition:]
                else:
                    cond_x = None

                z = torch.randn_like(x)

                # #### Changed - New loss computation ###
                # Old: Used distillation loss with teacher model
                # optim.zero_grad()
                # loss = self.loss(x, z, t0, t1, ema_model=ema_model, diffusion=model_diff, cond=cond_x, device=device)

                # New: Direct consistency training with optional multistep
                optim.zero_grad()
                if use_multistep and len(boundaries) > 3:
                    loss = self.multistep_loss(x, z, boundaries, cond=cond_x)
                else:
                    N = len(boundaries) - 1
                    t_idx = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
                    t0 = boundaries[t_idx]
                    t1 = boundaries[t_idx + 1]
                    loss = self.loss_ct(x, z, t0, t1, cond=cond_x)
                ### End of changed ####

                loss.backward()

                # #### Changed - Remove EMA updates for CT ###
                # Old: EMA model updates (only needed for CD)
                # with torch.no_grad():
                #     mu = math.exp(2 * math.log(0.95) / N)
                #     for p, ema_p in zip(self.parameters(), ema_model.parameters()):
                #         ema_p.mul_(mu).add_(p, alpha=1 - mu)

                # New: Simple gradient step (CT doesn't need EMA)
                ### End of changed ####

                optim.step()

                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.6f}, boundaries: {len(boundaries)}")
        return