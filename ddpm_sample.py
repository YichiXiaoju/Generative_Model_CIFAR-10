# sample_ddpm.py
import os
import torch
from torchvision.utils import save_image, make_grid

from models.unet import TinyUNet
from utils.schedules import make_beta_schedule_cosine as make_beta_schedule, compute_alphas



@torch.no_grad()
def p_sample(net, x, t_int, betas, alphas, alpha_bars):
    """
    DDPM smpling step.
    x: (B, C, H, W), noisy image at step t
    t_int: (B,), current step (integer)
    betas, alphas, alpha_bars: (T,), precomputed schedule
    return: x_prev, (B, C, H, W), less noisy image at step t-1
    """
    T = betas.shape[0]

    beta_t      = betas[t_int][:, None, None, None]
    alpha_t     = alphas[t_int][:, None, None, None]
    alpha_bar_t = alpha_bars[t_int][:, None, None, None]

    t_prev = torch.clamp(t_int - 1, min=0)
    alpha_bar_prev = alpha_bars[t_prev][:, None, None, None]

    # predict noise ε_theta(x_t, t)
    t_scalar = (t_int.float() + 1e-6) / (T - 1)
    eps = net(x, t_scalar)

    # calculate mean
    mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * eps) / torch.sqrt(alpha_t + 1e-8)

    # posterior variance beta_tilde
    posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)
    sigma = torch.sqrt(torch.clamp(posterior_var, min=1e-20))

    noise = torch.randn_like(x)
    nonzero = (t_int > 0).float()[:, None, None, None]
    x_prev = mean + nonzero * sigma * noise

    # for numerical stability
    x_prev = torch.nan_to_num(x_prev, nan=0.0, posinf=0.0, neginf=0.0)
    return x_prev


@torch.no_grad()
def sample_ddpm(model_path="runs/ddpm/latest_ema.pt", n=16, T=1000, device=None, out_path="samples/ddpm_16.png"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    net = TinyUNet(base=128, time_dim=128, out_channels=3).to(device).eval()
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)

    # setup noise schedule
    betas = make_beta_schedule(T).to(device)
    alphas, alpha_bars = compute_alphas(betas)

    # sanity check
    assert torch.all(alpha_bars[1:] <= alpha_bars[:-1]), "alpha_bars must be non-increasing!"

    # sampling
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    x = torch.randn(n, 3, 32, 32, device=device)

    for i in range(T - 1, -1, -1):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        x = p_sample(net, x, t, betas, alphas, alpha_bars)
        if i % 100 == 0 or i == T - 1:
            print(f"Sampling step {T - i}/{T}")

    # save images
    x = (x.clamp(-1, 1) + 1) * 0.5
    grid = make_grid(x, nrow=int(n ** 0.5))
    save_image(grid, out_path)
    print(f"✅ saved to {out_path}")


if __name__ == "__main__":
    sample_ddpm(model_path="runs/ddpm/latest_ema.pt", n=16, T=1000, out_path="samples/ddpm_16.png")