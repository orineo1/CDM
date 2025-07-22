# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import trange
# from NN_utils import GenericNN,TimeEmbedding  # Assuming you've defined this elsewhere
#
# class DiffusionModelIncludingCond(nn.Module):
#     def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
#                  condition_on: int = 1, diffusion_steps: int = 100, nblocks: int = 6, device=None):
#         super(DiffusionModel, self).__init__()
#         self.condition = condition
#         self.nfeatures = nfeatures
#         self.nunits = nunits
#         self.diffusion_steps = diffusion_steps
#         self.condition_on = condition_on
#         self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if condition:
#             self.cond_embed = nn.Linear(condition_on, nunits)
#
#         self.inblock = nn.Linear(nfeatures, nunits)
#         self.out = nn.Linear(nunits, nfeatures)
#         self.time_embed = TimeEmbedding(nunits)
#         self.net = GenericNN(nunits, [nunits] * nblocks, nunits).to(self.device)
#
#         self.beta_0 = 0.1
#         self.beta_1 = 20.0
#         self.to(self.device)
#
#     def log_alpha(self, t):
#         return -0.5 * t * self.beta_0 - 0.25 * t ** 2 * (self.beta_1 - self.beta_0)
#
#     def dlog_alphadt(self, t):
#         return -0.5 * self.beta_0 - 0.5 * t * (self.beta_1 - self.beta_0)
#
#     def log_sigma(self, t):
#         return torch.log(t)
#
#     def dlog_sigmadt(self, t):
#         return 1.0 / t
#
#     def beta_func(self, t):
#         return 1 + 0.5 * t * self.beta_0 + 0.5 * t ** 2 * (self.beta_1 - self.beta_0)
#
#     def q_t(self, data, t):
#         data = data.to(self.device)
#         t = t.to(self.device)
#         eps = torch.randn_like(data, device=self.device)
#         log_a = self.log_alpha(t).to(self.device)
#         log_s = self.log_sigma(t).to(self.device)
#         x_t = torch.exp(log_a) * data + torch.exp(log_s) * eps
#         return eps, x_t
#
#     def sm_loss(self, x, cond, t, eps, x_t):
#         pred_eps = self(t, x_t, cond)
#         return torch.mean((eps + pred_eps) ** 2)
#
#     def forward(self, t, x, cond=None):
#         x = x.to(self.device)
#         t = t.to(self.device)
#         t_emb = self.time_embed(t)
#
#         x_proj = self.inblock(x)
#         if self.condition and cond is not None:
#             cond_emb = self.cond_embed(cond.to(self.device))
#             val = x_proj + cond_emb
#         else:
#             val = x_proj
#
#         val = self.net(val, t_emb)
#         return self.out(val)
#         # return self.out(val) + x  # residual connection
#
#     def vector_field_SDE(self, t, x, cond=None,track_gradients=False):
#         context = torch.no_grad() if not track_gradients else torch.enable_grad()
#         with context:
#             out = self(t, x, cond)
#             dxdt = self.dlog_alphadt(t) * x - 2 * self.beta_func(t) * out
#         return dxdt
#
#
#     def sample(self, nsamples=512, cond_val=None, track_gradients=False):
#         self.eval()
#         dt = 1e-2
#         t = 1.0
#         n = int(t / dt)
#
#         x = torch.randn(nsamples, self.nfeatures, device=self.device, requires_grad=track_gradients)
#         samples = [x]
#
#         if self.condition and cond_val is not None:
#             if cond_val.dim() == 1:
#                 cond = cond_val.unsqueeze(0).repeat(nsamples, 1)
#             elif cond_val.dim() == 2 and cond_val.shape[0] == nsamples:
#                 cond = cond_val
#             else:
#                 raise ValueError(f"cond_val shape invalid: {cond_val.shape}")
#
#         current_t = torch.full((nsamples, 1), t, device=self.device)
#
#         context = torch.enable_grad() if track_gradients else torch.no_grad()
#         with context:
#             for _ in range(n):
#                 noise = torch.randn(nsamples, self.nfeatures, device=self.device)
#
#                 dx = (
#                         -dt * self.vector_field_SDE(current_t, samples[-1], cond)
#                         + torch.sqrt(2 * torch.exp(self.log_sigma(current_t)) * self.beta_func(current_t) * dt) * noise
#                 )
#                 x_next = samples[-1] + dx  # Not in-place
#                 samples.append(x_next)
#                 current_t -= dt
#
#         x_gen = torch.stack(samples, dim=1)  # [nsamples, n+1, nfeatures]
#         return x_gen
#
#     def train_model(self, data_generator=None, dataloader=None, num_iterations=5000):
#         assert data_generator or dataloader, "Need either data_generator or dataloader"
#
#         optimizer = optim.Adam(self.parameters(), lr=2e-4)
#         bs = 512
#         progress_bar = trange(num_iterations)
#         dataloader_iter = iter(dataloader) if dataloader else None
#
#         for step in progress_bar:
#             self.train()
#             optimizer.zero_grad()
#
#             if dataloader_iter:
#                 try:
#                     batch = next(dataloader_iter)
#                 except StopIteration:
#                     dataloader_iter = iter(dataloader)
#                     batch = next(dataloader_iter)
#
#                 x = batch[0].to(self.device).float()
#                 cond = batch[1].to(self.device).float() if self.condition and len(batch) > 1 else x[:, :self.condition_on]
#             else:
#                 x = data_generator(bs, device=self.device)
#                 cond = x[:, :self.condition_on] if self.condition else None
#
#             t = torch.rand(x.shape[0], 1, device=self.device)
#             eps, x_t = self.q_t(x, t)
#             loss = self.sm_loss(x, cond, t, eps, x_t)
#             loss.backward()
#             optimizer.step()
#             progress_bar.set_postfix(loss=loss.item())
#
#         return self
#
#     def sample_ddim_step(self, x_t, t, cond=None, dt=1e-2):
#
#         pred_eps = self(t, x_t, cond)
#
#         log_alpha_t = self.log_alpha(t)
#         log_sigma_t = self.log_sigma(t)
#
#         exp_log_alpha = torch.exp(log_alpha_t)
#         exp_log_sigma = torch.exp(log_sigma_t)
#
#         x0_hat = (x_t - exp_log_sigma * pred_eps) / exp_log_alpha
#         t_prev = t - dt
#         log_alpha_prev = self.log_alpha(t_prev)
#         log_sigma_prev = self.log_sigma(t_prev)
#
#         x_prev = (torch.exp(log_alpha_prev) * x0_hat +
#                   torch.exp(log_sigma_prev) * pred_eps)
#
#         return x_prev, x0_hat


#
# ########################### VERSION 2 - without the conditional #####################################
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from NN_utils import GenericNN, TimeEmbedding  # Assuming you've defined this elsewhere


class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
                 condition_on: int = 1, diffusion_steps: int = 100, nblocks: int = 6, device=None):
        super(DiffusionModel, self).__init__()
        self.condition = condition
        self.nfeatures = nfeatures
        # change
        self.gen_features = nfeatures - condition_on if condition else nfeatures
        # changed
        self.nunits = nunits
        self.diffusion_steps = diffusion_steps
        self.condition_on = condition_on
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if condition:
            self.cond_embed = nn.Linear(condition_on, nunits)

        # change
        self.inblock = nn.Linear(self.gen_features, nunits)
        self.out = nn.Linear(nunits, self.gen_features)
        # changed
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
        # change
        if self.condition:
            # Only add noise to the non-conditioning features
            gen_data = data[:, self.condition_on:]
        else:
            gen_data = data

        eps = torch.randn_like(gen_data, device=self.device)
        # changed
        log_a = self.log_alpha(t).to(self.device)
        log_s = self.log_sigma(t).to(self.device)
        # change
        x_t = torch.exp(log_a) * gen_data + torch.exp(log_s) * eps
        # changed
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

    def vector_field_SDE(self, t, x, cond=None, track_gradients=False):
        context = torch.no_grad() if not track_gradients else torch.enable_grad()
        with context:
            out = self(t, x, cond)
            dxdt = self.dlog_alphadt(t) * x - 2 * self.beta_func(t) * out
        return dxdt

    def sample(self, nsamples=512, cond_val=None, track_gradients=False):
        self.eval()
        dt = 1e-2
        t = 1.0
        n = int(t / dt)

        # change
        x = torch.randn(nsamples, self.gen_features, device=self.device, requires_grad=track_gradients)
        # changed
        samples = [x]

        if self.condition and cond_val is not None:
            if cond_val.dim() == 1:
                cond = cond_val.unsqueeze(0).repeat(nsamples, 1)
            elif cond_val.dim() == 2 and cond_val.shape[0] == nsamples:
                cond = cond_val
            else:
                raise ValueError(f"cond_val shape invalid: {cond_val.shape}")

        current_t = torch.full((nsamples, 1), t, device=self.device)

        context = torch.enable_grad() if track_gradients else torch.no_grad()
        with context:
            for _ in range(n):
                # change
                noise = torch.randn(nsamples, self.gen_features, device=self.device)
                # changed

                dx = (
                        -dt * self.vector_field_SDE(current_t, samples[-1], cond)
                        + torch.sqrt(2 * torch.exp(self.log_sigma(current_t)) * self.beta_func(current_t) * dt) * noise
                )
                x_next = samples[-1] + dx  # Not in-place
                samples.append(x_next)
                current_t -= dt

        x_gen = torch.stack(samples, dim=1)  # [nsamples, n+1, nfeatures]

        # change
        if self.condition and cond_val is not None:
            # Concatenate conditioning values to the generated samples
            cond_expanded = cond.unsqueeze(1).expand(-1, x_gen.shape[1], -1)  # [nsamples, n+1, condition_on]
            x_gen = torch.cat([cond_expanded, x_gen], dim=2)  # [nsamples, n+1, nfeatures]
        # changed

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
                cond = batch[1].to(self.device).float() if self.condition and len(batch) > 1 else x[:,
                                                                                                  :self.condition_on]
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

    def sample_ddim_step(self, x_t, t, cond=None, dt=1e-2):

        pred_eps = self(t, x_t, cond)

        log_alpha_t = self.log_alpha(t)
        log_sigma_t = self.log_sigma(t)

        exp_log_alpha = torch.exp(log_alpha_t)
        exp_log_sigma = torch.exp(log_sigma_t)

        x0_hat = (x_t - exp_log_sigma * pred_eps) / exp_log_alpha
        t_prev = t - dt
        log_alpha_prev = self.log_alpha(t_prev)
        log_sigma_prev = self.log_sigma(t_prev)

        x_prev = (torch.exp(log_alpha_prev) * x0_hat +
                  torch.exp(log_sigma_prev) * pred_eps)

        return x_prev, x0_hat