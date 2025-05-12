
import importlib
import Diffusion
from LossFunctions import MMDLoss,RBF
from dist_utils import generate_mog_samples
import ConsistencyModels
import  FlowMatching
from ConsistencyModels import ConsistencyModel
import dist_utils
from tqdm import tqdm
import torch
import torch.optim as optim_cond

######### Diffusion

#### Optimize LGD
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights,mu_list, Sigma_list, alpha, nsamples=250, num_x_t=3, loss="MMD"):
    mmd_loss =MMDLoss(kernel=RBF())


    # Initialize starting point
    x_t = torch.zeros(model_uncond.nfeatures, requires_grad=True)

    # Loop over diffusion steps
    pbar= tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))
    # Generate samples from Mixture of Gaussians
    samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0)

    for t in pbar:
        # Create a fresh tensor at each step to ensure gradient independence
        x_t = x_t.detach().clone().requires_grad_(True)  # TODO change this into vector of the size of the condition by of the uncond
        # Create a new optimizer for this step's tensor
        optimizer = optim_cond.Adam([x_t], lr=0.05)

        # Perform one DDIM step
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device="cpu", eta=0.0) # TODO use sample_ddim instead

        current_var = model_cond.betas[t]

        # Zero gradients before computing losses
        optimizer.zero_grad()

        sum_exp_loss = 0
        for j in range(num_x_t):

            # current_var of x_0_cont_xt_hat
            sigma_x_0_cond_xt = torch.sqrt(current_var) / torch.sqrt(1 + current_var)

            # Sample from multivariate normal N(x_t, cov_matrix)
            noise = torch.randn_like(x_t)  # Standard normal noise
            x_0_cont_xt_hat = x_t + sigma_x_0_cond_xt * noise
            # Sample from Y_0 | X_0 = pred_x0
            condition = x_0_cont_xt_hat.view(1, -1).repeat(nsamples, 1)
            if loss=="MMD":
                target_samples, _ = model_cond.sample_ddim(nsamples=nsamples, condition_x=condition, t_start=None, t_end=None)

                target_samples = target_samples[:, model_cond.condition_on:]  # Extract relevant parts



                # Compute MMD loss for this sample
                batch_loss = mmd_loss(target_samples, samples)


            sum_exp_loss += torch.exp(-batch_loss)

        # Compute mean loss and backpropagate
        mean_loss = -torch.log(sum_exp_loss/ num_x_t)
        mean_loss.backward()

        # Get the gradient from x_t
        grad = x_t.grad.clone()


        # Update x manually using the computed gradient
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - grad
        pbar.set_description(f"Step {t} | x_t: {x_t.item():.3f}")

    x_t = x_t.reshape(-1, 1)
    return x_t