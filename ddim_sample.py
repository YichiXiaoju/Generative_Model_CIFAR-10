import os, torch
from torchvision.utils import save_image, make_grid
from models.unet import TinyUNet
from utils.schedules import make_beta_schedule_cosine as make_beta_schedule, compute_alphas
from methods.ddim import ddim_sample

def load_unet_from_ckpt(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    base = state["in_conv.weight"].shape[0]
    net = TinyUNet(base=base, time_dim=128, out_channels=3).to(device).eval()
    net.load_state_dict(state, strict=True)
    return net

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "runs/ddpm/latest_ema.pt"     # default to EMA weights
    # ckpt = "runs/ddpm/latest.pt"        # uncomment to try non-EMA weights
    net = load_unet_from_ckpt(ckpt, device)

    T = 1000
    betas = make_beta_schedule(T).to(device)   
    _, alpha_bars = compute_alphas(betas)

    imgs = ddim_sample(net, alpha_bars, n=16, steps=50, eta=0.0, device=device)
    imgs = (imgs.clamp(-1,1) + 1) * 0.5
    os.makedirs("samples", exist_ok=True)
    grid = make_grid(imgs, nrow=4)
    save_image(grid, "samples/ddim_50.png")
    print("✅ saved samples/ddim_50.png")
