
import importlib
import Diffusion
from LossFunctions import MMDLoss,RBF
from dist_utils import generate_mog_samples
import ConsistencyModels
import  FlowMatching
from ConsistencyModels import ConsistencyModel
import dist_utils
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.optim as optim_cond

######### Diffusion
def optimize_DPS(model_uncond, model_cond, mog_means, mog_variances, weights,mu_list, Sigma_list, alpha, nsamples=250, num_x_t=3, loss="MMD",CM=False):
    mmd_loss =MMDLoss(kernel=RBF())


    # Initialize starting point
    x_t= torch.randn(model_uncond.nfeatures, requires_grad=True )
    # print(f"x_t: {x_t}")
    # Loop over diffusion steps
    pbar= tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))
    for t in pbar:
        # Create a fresh tensor at each step to ensure gradient independence
        x_t = x_t.detach().clone().requires_grad_(True)
        # Create a new optimizer for this step's tensor
        optimizer = optim_cond.Adam([x_t], lr=0.05)

        # Perform one DDIM step
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device="cpu", eta=0.0)

        # Sample from Y_0 | X_0 = pred_x0
        condition = pred_x0.view(1, -1).repeat(nsamples, 1)
        target_samples, _ = model_cond.sample(nsamples=nsamples, condition_x=condition)
        if not CM:
            target_samples = target_samples[:, model_cond.condition_on:]  # Extract relevant parts

        # Generate samples from Mixture of Gaussians
        samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0)

        # Zero gradients before computing loss
        optimizer.zero_grad()

        # Compute MMD loss
        loss = mmd_loss(target_samples, samples)

        # Compute gradients
        loss.backward()

        # Extract gradient and scale by variance (if needed)
        current_var = model_uncond.betas[t]
        grad = x_t.grad.clone() * current_var
        pbar.set_description(f"Step {t} | pred_x0: {pred_x0}| mean_loss: {loss.item():.3f}")

        # Update x manually using the computed gradient
        # Instead of using the optimizer's step, we're applying our custom update
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - grad
    x_t = x_t.reshape(-1, 1)
    return x_t


#### Optimize LGD
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights,mu_list, Sigma_list, alpha, nsamples=250, num_x_t=3, loss="MMD",CM=False):
    mmd_loss =MMDLoss(kernel=RBF())

    # Initialize starting point
    x_t= torch.randn(model_uncond.nfeatures, requires_grad=True )
    # Loop over diffusion steps
    pbar= tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))
    # Generate samples from Mixture of Gaussians
    samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0)
    for t in pbar:
        # Create a fresh tensor at each step to ensure gradient independence
        x_t = x_t.detach().clone().requires_grad_(True)
        # Create a new optimizer for this step's tensor
        optimizer = optim_cond.Adam([x_t], lr=0.05)

        # Perform one DDIM step
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device="cpu", eta=0.0) # TODO use sample_ddim instead
        current_var = model_uncond.betas[t]

        # Zero gradients before computing losses
        optimizer.zero_grad()

        sum_exp_loss = 0
        for j in range(num_x_t):
            # current_var of x_0_cont_xt_hat
            sigma_x_0_cond_xt =torch.sqrt(current_var)/torch.sqrt(1+torch.pow(current_var,2))

            # Sample from multivariate normal N(x_t, cov_matrix)
            noise = torch.randn_like(x_t)  # Standard normal noise
            x_0_cont_xt_hat =( x_t_minus_1 + sigma_x_0_cond_xt * noise)

            # Sample from Y_0 | X_0 = pred_x0
            condition = x_0_cont_xt_hat.view(1, -1).repeat(nsamples, 1)
            if loss=="MMD":
                target_samples, _ = model_cond.sample(nsamples=nsamples, condition_x=condition)
                if not CM:
                    target_samples = target_samples[:, model_cond.condition_on:]  # Extract relevant parts
                # Compute MMD loss for this sample
                batch_loss = mmd_loss(target_samples, samples)
                pbar.set_description(f"Step {t} | pred_x0_model: {pred_x0.item():.3f}| pred_x0: {x_0_cont_xt_hat.item():.3f}| mean_loss: {batch_loss.item():.3f}")

            sum_exp_loss += torch.exp(-batch_loss)

        # Compute mean loss and backpropagate
        mean_loss = -torch.log(sum_exp_loss/ num_x_t)
        mean_loss.backward()

        # Get the gradient from x_t
        grad = x_t.grad.clone()

        # Update x manually using the computed gradient
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - grad

    x_t = x_t.reshape(-1, 1)
    return x_t

######### FlowMatching

#### DFlow
def find_samples_to_kl(numb_points,mog_means,mog_variances,weights):
    # Convert mog_means (list of tensors) into a single tensor
    mog_means_tensor = torch.stack(mog_means, dim=0)  # Concatenates along the first axis

    # Generate MoG samples
    mog_samples = generate_mog_samples(numb_points, mog_means, mog_variances, weights)

    # Concatenate both mog_means and generated samples along dim=0
    Target = torch.cat([mog_means_tensor, mog_samples], dim=0)  # Concatenates along rows
    Target=Target.to(torch.float32)
    return Target


def compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method, mog_means, mog_variances, weights, device):
    if loss_method == "MMD":
        mmd_loss = MMDLoss(kernel=RBF())
        x_cond = x_1.reshape(-1) * torch.ones(
            samples.shape[0], vf_y_cond_x.condition_on, dtype=torch.float32, device=device
        )
        target_samples = vf_y_cond_x.sample(samples.shape[0], x_cond)
        loss = mmd_loss(target_samples, samples)
    elif loss_method == "KL":
        target = find_samples_to_kl(samples.shape[0], mog_means, mog_variances, weights)
        x_cond = x_1.reshape(-1) * torch.ones(
            target.shape[0], vf_y_cond_x.condition_on, dtype=torch.float32, device=device
        )
        log_prob_flow_cond = vf_y_cond_x.compute_likelihood(target, x_cond)
        log_prob_target = dist_utils.mog_multivariate_pdf(target, mog_means, mog_variances, weights=weights, log=True)
        loss = F.kl_div(log_prob_flow_cond, log_prob_target, reduction="batchmean", log_target=True)
    else:
        raise ValueError(f"Unknown loss method: {loss_method}")
    return loss

## TODO add regulation https://arxiv.org/pdf/2402.14017
def optimize_DFLOW(vf_y_cond_x, vf_X, device, mog_means, mog_variances, weights,
                   max_iter=100, FLAG=False, n_sample=50, loss_method="MMD"):
    x = torch.randn((1, vf_X.nfeatures), dtype=torch.float32, device=vf_X.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn="strong_wolfe")
    samples = generate_mog_samples(n_sample, mog_means, mog_variances, weights, tau=10.0)

    best_x, best_loss = None, float('inf')

    def closure():
        optimizer.zero_grad()
        x_1 = vf_X.sample(num_sample=1, y=None, x_init=x)

        loss = compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method,
                            mog_means, mog_variances, weights, device)

        nonlocal best_x, best_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.clone()

        if FLAG:
            print(f"x_0: {x.detach().cpu().numpy()} | x_1: {x_1} | loss: {loss}")

        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    best_x1 = vf_X.sample(num_sample=1, y=None, x_init=best_x)
    return best_x1, best_loss

