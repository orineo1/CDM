
import importlib
import Diffusion
from LossFunctions import MMDLoss,RBF
from dist_utils import generate_mog_samples
import ConsistencyModels
import  FlowMatching
from ConsistencyModels import ConsistencyModel
import dist_utils
import torch.nn.functional as F
import torch
import torch.optim as optim_cond
from tqdm.notebook import tqdm

######### Diffusion
def optimize_DPS(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda"):
    mmd_loss = MMDLoss(kernel=RBF())

    # Move initial tensor to CUDA and set requires_grad
    x_t = torch.randn(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.detach().clone().to(device).requires_grad_(True)
    optimizer = optim_cond.Adam([x_t], lr=0.05)
    # Move mog distribution parameters to device as well, if they are tensors
    if isinstance(mog_means, torch.Tensor):
        mog_means = mog_means.to(device)

    if isinstance(mog_variances, torch.Tensor):
        mog_variances = mog_variances.to(device)

    if isinstance(weights, torch.Tensor):
        weights = weights.to(device)

    best_mmd_loss = float("inf")
    best_pred_x0 = None

    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))
    for t in pbar:


        # Make sure the model_uncond sampling runs on device
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)

        condition = pred_x0.view(1, -1).repeat(nsamples, 1).to(device)
        target_samples, _ = model_cond.sample(nsamples=nsamples, condition_x=condition,device=device)
        target_samples = target_samples.to(device)

        if not CM:
            target_samples = target_samples[:, model_cond.condition_on:]

        samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0).to(device)

        optimizer.zero_grad()
        loss_val = mmd_loss(target_samples, samples)
        loss_val.backward()

        current_var = model_uncond.betas[t].to(device)
        grad = x_t.grad.clone() * current_var

        pbar.set_description(
            f"Step {t} | pred_x0_model: {pred_x0} |  pred_x0 loss: {loss_val.item():.3f}"
        )

        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone().to(device) - grad
        x_t.requires_grad_()


        if loss_val.item() < best_mmd_loss:
            best_mmd_loss = loss_val.item()
            best_pred_x0 = pred_x0.detach().clone()

    x_t_final = x_t.detach().clone()

    del x_t,optimizer, grad, x_t_minus_1, pred_x0, condition, target_samples, loss_val
    torch.cuda.empty_cache() if device == "cuda" else None

    return x_t_final, best_pred_x0, best_mmd_loss


#### Optimize LGD

## Optimize LGD
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda"):

    mmd_loss = MMDLoss(kernel=RBF())
    best_mmd_loss = float("inf")
    best_x_0_cont_xt_hat = None
    # Initialize starting point
    x_t = torch.zeros(model_uncond.nfeatures, requires_grad=True)

    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))

    # Loop over diffusion steps
    for t in pbar:
        # Create a fresh tensor at each step to ensure gradient independence
        x_t = x_t.detach().clone().requires_grad_(True)  # TODO change this into vector of the size of the condition by of the uncond

        # Create a new optimizer for this step's tensor
        optimizer = optim_cond.Adam([x_t], lr=0.05)

        # Perform one DDIM step
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device="cpu", eta=0.0)


        current_var = model_uncond.betas[t].to(device)#model_cond.betas[t]

        # Zero gradients before computing losses
        optimizer.zero_grad()

        sum_exp_loss = 0
        for j in range(num_x_t):
            # Get the dimension of x_t
            dim = x_t.shape[-1]  # This will be 24 in your case

            # current_var of x_0_cont_xt_hat
            sigma_x_0_cond_xt = torch.sqrt(current_var) / torch.sqrt(1 + current_var)

            # Sample from multivariate normal N(x_t, cov_matrix)
            noise = torch.randn_like(x_t)  # Standard normal noise
            x_0_cont_xt_hat = x_t + sigma_x_0_cond_xt * noise
            # _, pred_x0_CM = model_uncond.sample_ddim_step(x_0_cont_xt_hat, t-1, condition_x=None, device="cpu", eta=0.0)

            # Sample from Y_0 | X_0 = pred_x0
            condition = x_0_cont_xt_hat.view(1, -1).repeat(nsamples, 1)
            if loss=="MMD":
                target_samples, _ = model_cond.sample(nsamples=nsamples,  condition_x=condition,device=device)

                if not CM:
                    target_samples = target_samples[:, model_cond.condition_on:]

                # Generate samples from Mixture of Gaussians
                samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0)

                # Compute MMD loss for this sample
                batch_loss = mmd_loss(target_samples, samples)
                if batch_loss.item() < best_mmd_loss:
                    best_mmd_loss = batch_loss.item()
                    best_x_0_cont_xt_hat = x_0_cont_xt_hat.detach().clone()
                pbar.set_description(
                    f"Step {t} | xt :{x_t}| pred_x0_model: {pred_x0} | pred_x0_CM: {x_0_cont_xt_hat} | batch_loss: {batch_loss.item():.3f}"
                )

            sum_exp_loss += torch.exp(-batch_loss)

        # Compute mean loss and backpropagate
        mean_loss = -torch.log(sum_exp_loss/ num_x_t)
        mean_loss.backward()

        # Get the gradient from x_t
        grad = x_t.grad.clone()

        # Update x manually using the computed gradient
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - grad
    # x_t = x_t.reshape(-1, 1)
    # return x_t, x_t, 2
    x_t_final = x_t.detach().clone()
    del x_t,optimizer, grad, x_t_minus_1, pred_x0, condition, target_samples, x_0_cont_xt_hat, mean_loss, batch_loss
    torch.cuda.empty_cache() if device == "cuda" else None
#
    return x_t_final, x_t_final, best_mmd_loss


# def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
#                  nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda"):
#     mmd_loss = MMDLoss(kernel=RBF())
#
#     # Initialize x_t on device with requires_grad
#     x_t = torch.randn(model_uncond.nfeatures, device=device, requires_grad=True)
#     x_t = x_t.detach().clone().to(device).requires_grad_(True)
#     optimizer = optim_cond.Adam([x_t], lr=0.05)
#
#     if isinstance(mog_means, torch.Tensor):
#         mog_means = mog_means.to(device)
#
#     if isinstance(mog_variances, torch.Tensor):
#         mog_variances = mog_variances.to(device)
#
#     if isinstance(weights, torch.Tensor):
#         weights = weights.to(device)
#
#     best_mmd_loss = float("inf")
#     best_x_0_cont_xt_hat = None
#
#     pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))
#
#     # Pre-generate samples on device once before loop
#     samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0).to(device)
#     for t in pbar:
#
#         # Sample on the correct device
#         x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
#         current_var = model_uncond.betas[t].to(device)
#
#         optimizer.zero_grad()
#         sum_exp_loss = 0
#
#         for j in range(num_x_t):
#             sigma_x_0_cond_xt = torch.sqrt(current_var) / torch.sqrt(1 + torch.pow(current_var, 2))
#             noise = torch.randn_like(x_t)
#
#             x_0_cont_xt_hat = x_t_minus_1 + sigma_x_0_cond_xt * noise
#             condition = x_0_cont_xt_hat.view(1, -1).repeat(nsamples, 1).to(device)
#
#             if loss == "MMD":
#                 target_samples, _ = model_cond.sample(nsamples=nsamples, condition_x=condition,device=device)
#                 target_samples = target_samples.to(device)
#
#                 if not CM:
#                     target_samples = target_samples[:, model_cond.condition_on:]
#
#                 batch_loss = mmd_loss(target_samples, samples)
#
#                 if batch_loss.item() < best_mmd_loss:
#                     best_mmd_loss = batch_loss.item()
#                     best_x_0_cont_xt_hat = x_0_cont_xt_hat.detach().clone()
#
#                 pbar.set_description(
#                     f"Step {t} | pred_x0_model: {pred_x0} | pred_x0: {x_0_cont_xt_hat} | batch_loss: {batch_loss.item():.3f}"
#                 )
#
#             sum_exp_loss += torch.exp(-batch_loss)
#
#         mean_loss = -torch.log(sum_exp_loss / num_x_t)
#
#         mean_loss.backward()
#         grad = x_t.grad.clone()
#
#         with torch.no_grad():
#             x_t = x_t_minus_1.detach().clone().to(device) - grad
#         x_t.requires_grad_()
#     x_t_final = x_t.detach().clone()
#
#     del x_t,optimizer, grad, x_t_minus_1, pred_x0, condition, target_samples, x_0_cont_xt_hat, mean_loss, batch_loss
#     torch.cuda.empty_cache() if device == "cuda" else None
#
#     return x_t_final, best_x_0_cont_xt_hat, best_mmd_loss




######### FlowMatching

#### DFlow
def find_samples_to_kl(numb_points, mog_means, mog_variances, weights, device="cuda"):
    if isinstance(mog_means, (list, tuple)):
        mog_means_tensor = torch.stack(mog_means, dim=0)
    elif isinstance(mog_means, torch.Tensor):
        mog_means_tensor = mog_means.unsqueeze(0) if mog_means.dim() == 0 else mog_means
    else:
        raise TypeError("mog_means must be a list, tuple, or tensor")

    if mog_means_tensor.dim() > 2 and mog_means_tensor.shape[-1] == 1:
        mog_means_tensor = mog_means_tensor.squeeze(-1)

    mog_means_tensor = mog_means_tensor.to(device)
    mog_samples = generate_mog_samples(numb_points, mog_means, mog_variances, weights).to(device)

    if mog_samples.dim() < mog_means_tensor.dim():
        mog_means_tensor = mog_means_tensor.squeeze(-1)

    Target = torch.cat([mog_means_tensor, mog_samples], dim=0).to(torch.float32)
    return Target



def compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method, mog_means, mog_variances, weights, device):
    if loss_method == "MMD":
        mmd_loss = MMDLoss(kernel=RBF())

        # Create conditioning tensor on device
        x_cond = x_1.reshape(-1).to(device) * torch.ones(
            samples.shape[0], vf_y_cond_x.condition_on, dtype=torch.float32, device=device
        )
        target_samples = vf_y_cond_x.sample(samples.shape[0], x_cond)
        target_samples = target_samples.to(device)
        samples = samples.to(device)

        loss = mmd_loss(target_samples, samples)

    elif loss_method == "KL":
        target = find_samples_to_kl(samples.shape[0], mog_means, mog_variances, weights, device=device)
        x_cond = x_1.reshape(-1).to(device) * torch.ones(
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
                   max_iter=100, FLAG=False, n_sample=50, loss_method="MMD",line_search_fn=None):
    # Initialize x on the correct device with requires_grad
    x = torch.randn((1, vf_X.nfeatures), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn=line_search_fn)

    # Generate MoG samples on device
    samples = generate_mog_samples(n_sample, mog_means, mog_variances, weights, tau=10.0).to(device)

    best_x, best_loss = None, float('inf')

    def closure():
        optimizer.zero_grad()

        x_1 = vf_X.sample(num_sample=1, y=None, x_init=x)

        loss = compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method,
                            mog_means, mog_variances, weights, device)

        nonlocal best_x, best_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()

        if FLAG:
            print(f"x_0: {x.detach().cpu().numpy()} | x_1: {x_1} | loss: {loss.item():.6f}")

        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    best_x1 = vf_X.sample(num_sample=1, y=None, x_init=best_x.to(device))
    del optimizer, samples, x, best_x
    torch.cuda.empty_cache() if device == "cuda" else None

    return best_x1, best_loss


