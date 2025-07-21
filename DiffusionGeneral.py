import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from NN_utils import GenericNN,TimeEmbedding  # Assuming you've defined this elsewhere


class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
                 condition_on: int = 1, diffusion_steps: int = 100, nblocks: int = 6, device=None):
        super(DiffusionModel, self).__init__()
        self.condition = condition
        self.nfeatures = nfeatures
        self.nunits = nunits
        self.diffusion_steps = diffusion_steps
        self.condition_on = condition_on
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if condition:
            self.cond_embed = nn.Linear(condition_on, nunits)

        self.inblock = nn.Linear(nfeatures, nunits)
        self.out = nn.Linear(nunits, nfeatures)
        self.time_embed = TimeEmbedding(nunits)
        self.net = GenericNN(nunits, [nunits] * nblocks, nunits).to(self.device)

        self.beta_0 = 0.1
        self.beta_1 = 20.0
        self.to(self.device)

    def log_alpha(self, t):
        return -0.5 * t * self.beta_0 - 0.25 * t ** 2 * (self.beta_1 - self.beta_0)

    def dlog_alphadt(self, t):
        return -0.5 * self.beta_0 - 0.5 * t * (self.beta_1 - self.beta_0)

    def log_sigma(self, t):
        return torch.log(t)

    def dlog_sigmadt(self, t):
        return 1.0 / t

    def beta_func(self, t):
        return 1 + 0.5 * t * self.beta_0 + 0.5 * t ** 2 * (self.beta_1 - self.beta_0)

    def q_t(self, data, t):
        data = data.to(self.device)
        t = t.to(self.device)
        eps = torch.randn_like(data, device=self.device)
        log_a = self.log_alpha(t).to(self.device)
        log_s = self.log_sigma(t).to(self.device)
        x_t = torch.exp(log_a) * data + torch.exp(log_s) * eps
        return eps, x_t

    def sm_loss(self, x, cond, t, eps, x_t):
        pred_eps = self(t, x_t, cond)
        return torch.mean((eps + pred_eps) ** 2)

    def forward(self, t, x, cond=None):
        x = x.to(self.device)
        t = t.to(self.device)
        t_emb = self.time_embed(t)

        x_proj = self.inblock(x)
        if self.condition and cond is not None:
            cond_emb = self.cond_embed(cond.to(self.device))
            val = x_proj + cond_emb
        else:
            val = x_proj

        val = self.net(val, t_emb)
        return self.out(val)
        # return self.out(val) + x  # residual connection

    def vector_field_SDE(self, t, x, cond=None):
        with torch.no_grad():
            out = self(t, x, cond)
            dxdt = self.dlog_alphadt(t) * x - 2 * self.beta_func(t) * out
        return dxdt

    def generate_samples(self, bs=512, cond_val=None):
        self.eval()
        dt = 1e-2
        t = 1.0
        n = int(t / dt)
        x_gen = torch.zeros(bs, n + 1, self.nfeatures, device=self.device)
        x_gen[:, 0, :] = torch.randn(bs, self.nfeatures, device=self.device)

        cond = None
        if self.condition and cond_val is not None:
            cond = cond_val.unsqueeze(0).repeat(bs, 1)

        current_t = torch.full((bs, 1), t, device=self.device)

        with torch.no_grad():
            for i in trange(n):
                noise = torch.randn(bs, self.nfeatures, device=self.device)
                dx = (
                    -dt * self.vector_field_SDE(current_t, x_gen[:, i, :], cond)
                    + torch.sqrt(2 * torch.exp(self.log_sigma(current_t)) * self.beta_func(current_t) * dt) * noise
                )
                x_gen[:, i + 1, :] = x_gen[:, i, :] + dx
                current_t -= dt

        return x_gen

    def train_model(self, data_generator=None, dataloader=None, num_iterations=5000):
        assert data_generator or dataloader, "Need either data_generator or dataloader"

        optimizer = optim.Adam(self.parameters(), lr=2e-4)
        bs = 512
        progress_bar = trange(num_iterations)
        dataloader_iter = iter(dataloader) if dataloader else None

        for step in progress_bar:
            self.train()
            optimizer.zero_grad()

            if dataloader_iter:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                x = batch[0].to(self.device).float()
                cond = batch[1].to(self.device).float() if self.condition and len(batch) > 1 else x[:, :self.condition_on]
            else:
                x = data_generator(bs, device=self.device)
                cond = x[:, :self.condition_on] if self.condition else None

            t = torch.rand(x.shape[0], 1, device=self.device)
            eps, x_t = self.q_t(x, t)
            loss = self.sm_loss(x, cond, t, eps, x_t)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        return self
