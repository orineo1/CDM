import torch
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from torch import nn, Tensor
from tqdm import tqdm
import time
from torch.distributions import Independent, Normal
from FM_Solver_Extension_module import add_compute_likelihood_allow_grad
import math
# Activation class

from NN_utils import GenericNN,TimeEmbedding  # Assuming you've defined this elsewhere


class FMModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 time_dim: int,
                 y_dim: int,
                 hidden_dim: int,
                 depth: int = 4,
                 use_resnet: bool = True,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = False,
                 device: torch.device = None):

        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_resnet = use_resnet
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.condition = True if y_dim>0 else False


        total_input_dim = input_dim
        self.inblock = nn.Linear(total_input_dim, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)

        if y_dim>0:
            self.cond_embed = nn.Linear(y_dim, hidden_dim)
        self.vf = GenericNN(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim] * depth,  # depth controls number of residual layers
            output_dim=input_dim,
        )


        # Make sure vf is on the right device
        self.vf = self.vf.to(self.device)

        # instantiate an affine path object - ensure it's using the right device
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        # self.solver = ODESolver(velocity_model=self.forward)
        self.solver = ODESolver(velocity_model=self._wrapped_forward)

        # Move the entire model to the device
        self.to(self.device)

    def _wrapped_forward(self, t, x, y=None):
        return self.forward(x, t, x_cond=y)

    def forward(self,  x: Tensor, t: Tensor, x_cond: Tensor=None) -> Tensor:
        x_proj = self.inblock(x)
        t_emb = self.time_embed(t)

        if self.condition and x_cond is not None:
            cond_emb = self.cond_embed(x_cond)
            val = x_proj + cond_emb
        else:
            val = x_proj

        return self.vf(val, t_emb)

    def train_FM(self, lr=0.001, batch_size=1024, print_every=20000,
                 X=None, nepochs=10, max_grad_norm=None, mu_list=None, Sigma_list=None, alpha=None, Flag=False,
                 convergence_tol=1e-5, patience=3):
        # init optimizer
        # optim_FM = torch.optim.Adam(self.vf.parameters(), lr=lr)
        optim_FM = torch.optim.Adam(self.parameters(), lr=lr)

        use_loader = X is not None
        if use_loader:
            X = X.to(self.device)
            dataset = torch.utils.data.TensorDataset(X)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )

        epoch_range = range(nepochs)
        if Flag:
            epoch_range = tqdm(epoch_range)

        prev_loss = float('inf')
        stable_epochs = 0
        pbar = tqdm(epoch_range)
        for epoch in pbar:
            epoch_loss = 0.0
            num_batches = 0

            data_iter = dataloader if use_loader else range(1)

            for batch_idx in data_iter:
                optim_FM.zero_grad()

                if use_loader:
                    (batch_x,) = batch_idx
                else:
                    batch_x = dist_utils.generate_mog_samples(batch_size, mu_list, Sigma_list, alpha)

                x_1 = batch_x.to(self.device)
                x_x_1 = x_1[:, self.y_dim:].to(torch.float32).to(self.device)
                y_1 = x_1[:, :self.y_dim].reshape(-1, 1).to(torch.float32).to(self.device)
                x_0 = torch.randn_like(x_x_1).to(self.device)
                t = torch.rand(x_x_1.shape[0], device=self.device)

                path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_x_1)
                path_sample.x_t = path_sample.x_t.to(self.device)
                path_sample.dx_t = path_sample.dx_t.to(self.device)
                path_sample.t = path_sample.t.to(self.device)

                # loss = torch.pow(self.vf(path_sample.x_t, path_sample.t, y_1) - path_sample.dx_t, 2).mean()
                loss = torch.pow(self.forward(path_sample.x_t, path_sample.t, y_1) - path_sample.dx_t, 2).mean()

                epoch_loss += loss.item()
                num_batches += 1

                loss.backward()
                optim_FM.step()

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            pbar.set_description(f"loss: {avg_epoch_loss :.6f}")


            # Check convergence
            if abs(prev_loss - avg_epoch_loss) < convergence_tol:
                stable_epochs += 1
                if stable_epochs >= patience:
                    if Flag:
                        print(f"Converged after {epoch + 1} epochs.")
                    break
            else:
                stable_epochs = 0

            prev_loss = avg_epoch_loss

        # self.solver = ODESolver(velocity_model=self.vf)
        self.solver = ODESolver(velocity_model=self._wrapped_forward)
        from FM_Solver_Extension_module import add_compute_likelihood_allow_grad
        self.solver = add_compute_likelihood_allow_grad(self.solver)
        return

    def sample(self, num_sample, y=None, return_intermediates=False, step_size=0.05, x_init=None):
        """
        Sample from the vector field model with conditioning.

        Args:
            num_sample: Number of samples to generate
            y: Conditioning value(s). Can be:
               - A scalar that will be expanded to all samples
               - A tensor of shape [num_sample, 1] with different values for each sample
               - A tensor of shape [1, 1] that will be expanded
            return_intermediates: Whether to return intermediate steps
            step_size: Step size for the solver
            x_init: Initial values (if None, will use random normal)

        Returns:
            Samples from the model
        """
        T = torch.linspace(0, 1, 10)


        if y is not None:
            # Check shape and expand if needed
            if y.numel() == 1:
                # Single value: expand to all samples
                y = y * torch.ones(num_sample, dtype=torch.float32, device=self.device).reshape(-1, 1)
            elif len(y) == num_sample and (y.dim() == 1 or (y.dim() == 2 and y.shape[1] == 1)):
                # Already has correct shape [num_sample, 1] or [num_sample]
                y = y.reshape(num_sample, 1)
            else:
                raise ValueError(f"Incompatible y shape: {y.shape}. Expected scalar or tensor with shape [num_sample, 1]")

        # Initialize x values if needed
        if x_init is None:
            x_init = torch.randn((num_sample, self.input_dim), dtype=torch.float32, device=self.device)

        # Ensure x_init is on the correct device
        x_init = x_init.to(device=self.device)

        # Call the solver with the prepared inputs
        return self.solver.sample(
            time_grid=T,
            x_init=x_init,
            method='midpoint',
            step_size=step_size,
            return_intermediates=return_intermediates,
            y=y,
            enable_grad=True
        )

    def compute_likelihood(self, Target, y, step_size=0.05, enable_grad=True,
                           return_intermediates=False, return_log_det=False,
                           Train_consistency=False, exact_divergence=False,
                           time_grid=None):

        gaussian_log_density = Independent(Normal(torch.zeros(self.input_dim, device=self.device),
                                                  torch.ones(self.input_dim, device=self.device)), 1).log_prob

        # If no time grid is provided, create one with multiple points
        if time_grid is None and Train_consistency:
            # Create a time grid from 1.0 to 0.0 with small steps
            time_grid = torch.linspace(1.0, 0.0, 101, device=self.device)  # 101 points for 100 intervals
        elif time_grid is None:
            time_grid = torch.tensor([1.0, 0.0], device=self.device)

        if exact_divergence:
            sol, log_det = self.solver.compute_likelihood_allow_grad(
                x_1=Target, method='midpoint', step_size=step_size,
                exact_divergence=exact_divergence, log_p0=gaussian_log_density,
                y=y, enable_grad=enable_grad,
                Train_consistency=Train_consistency,
                time_grid=time_grid
            )
            return sol, log_det

        # For non-exact, average over multiple samples (as in your original code)
        num_acc = 20
        log_p_acc = 0

        for i in range(num_acc):
            sol, log_p = self.solver.compute_likelihood_allow_grad(
                x_1=Target, method='midpoint', step_size=step_size,
                exact_divergence=False, log_p0=gaussian_log_density,
                y=y, enable_grad=enable_grad,
                Train_consistency=Train_consistency,
                time_grid=time_grid
            )
            log_p_acc += log_p

        log_p_acc /= num_acc

        if Train_consistency:
            return sol, log_p
        return log_p_acc



