import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from NN_utils import GenericNN,TimeEmbedding


def kerras_boundaries(sigma_min, sigma_max, N, rho):
    steps = torch.arange(N + 1, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    return (sigma_max ** (-1 / rho) + steps / N * (sigma_min ** (-1 / rho) - sigma_max ** (-1 / rho))) ** (-rho)


class ConsistencyModel(nn.Module):
    def __init__(self, nfeatures: int, condition_on: int = 0, eps: float = 0.002, nunits: int = 128, depth: int = 6, device=None):
        super().__init__()
        self.eps = eps
        self.nfeatures = nfeatures
        self.condition_on = condition_on
        self.nunits = nunits
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = None

        if condition_on > 0:
            self.cond_embed = nn.Linear(condition_on, nunits)

        self.input_layer = nn.Linear(nfeatures, nunits)
        self.time_embed = TimeEmbedding(nunits)

        self.net = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] * depth,
            output_dim=nfeatures,
        )

        self.to(self.device)


    def forward(self, x, t, cond=None):
        x_ori = x
        x = self.input_layer(x)

        # Add conditioning if present
        if self.condition_on>0 and cond is not None:
            cond_emb = self.cond_embed(cond)
            x = x + cond_emb

        # Ensure t is (B, 1)
        if isinstance(t, (float, int)):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32, device=x.device).unsqueeze(1)
        time_emb = self.time_embed(t)

        x = self.net(x, time_emb)

        # Output skip connection with consistency weights
        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).sqrt()

        return c_skip_t * x_ori + c_out_t * x


    def loss(self, x, z, t0, t1, ema_model, diffusion=None, cond=None,device="cpu"):
        x2 = x + z * t1


        with torch.no_grad():
            if diffusion is not None:
                x1 = diffusion.sample_ddim_step( x2, t1, condition_x=cond, device=device, eta=0.0)
            else:
                x1 = x + z * t0
            x1 = ema_model(x1, t0, cond=cond)

        x2 = self(x2, t1, cond=cond)
        return F.mse_loss(x1, x2)


    def sample(self, nsamples=250, condition_x=None,ts: List[float]=[150.0, 50.0, 20.0, 10.0, 5.0,1.],device=None):
        device = self.device if device is None else device
        if condition_x is not None:
            condition_x = condition_x.to(device)

        x = torch.randn(nsamples, self.nfeatures, device=device) * ts[0]
        x = x.to(device)

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.eps ** 2) * z
            x = self(x, t, cond=condition_x)

        return x,None

    def train_model(self, X: torch.Tensor = None, nepochs: int = 100,batch_size:int=2048, device: str = "cpu", condition=None,
              model_diff=None):
        self.to(device)

        optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        ema_model = ConsistencyModel(nfeatures=self.nfeatures, condition_on=self.condition_on, nunits=self.nunits)
        ema_model.to(device)
        ema_model.load_state_dict(self.state_dict())

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pbar = tqdm(range(1, nepochs + 1))
        for epoch in pbar:
            N = math.ceil(math.sqrt((epoch * (150 ** 2 - 4) / nepochs) + 4) - 1) + 1
            boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)
            self.betas=boundaries
            loss_ema = None

            for (x,) in dataloader:
                x = x.to(device)

                if condition is not None and condition > 0:
                    assert condition < X.shape[1], "Condition dimension must be less than input dimension"
                    cond_x = x[:, :condition]
                    x = x[:, condition:]
                else:
                    cond_x = None

                z = torch.randn_like(x)
                t_idx = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
                t0 = boundaries[t_idx]
                t1 = boundaries[t_idx + 1]

                optim.zero_grad()
                loss = self.loss(x, z, t0, t1, ema_model=ema_model, diffusion=model_diff, cond=cond_x,device=device)
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
        return


