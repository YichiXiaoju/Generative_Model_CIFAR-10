# sample_rectified_flow.py
import os, torch
from torchvision.utils import save_image, make_grid
from models.unet import TinyUNet
from methods.rectified_flow import ode_step_euler, ode_step_heun

def load_unet_from_ckpt(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    base = state["in_conv.weight"].shape[0]
    net = TinyUNet(base=base, time_dim=128, out_channels=3).to(device).eval()
    net.load_state_dict(state, strict=True)
    return net, base

@torch.no_grad()
def sample_rectified_flow(
    model_path="runs/rectified_flow/latest_ema.pt",
    n=16, steps=100, solver="heun", out_path="samples/rf_100.png", device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    net, base = load_unet_from_ckpt(model_path, device)
    print(f"[RF] loaded '{model_path}' (base={base}), solver={solver}, steps={steps}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    x = torch.randn(n, 3, 32, 32, device=device)  # x(t=1)
    t_grid = torch.linspace(1.0, 0.0, steps+1, device=device)  # 1 → 0
    step_fn = ode_step_heun if solver.lower() == "heun" else ode_step_euler

    for i in range(steps):
        t, t_next = t_grid[i].item(), t_grid[i+1].item()
        dt = t_next - t 
        x = step_fn(net, x, t, dt)
        if i % max(1, steps//10) == 0:
            print(f"step {i}/{steps}: t={t:.3f}→{t_next:.3f}")

    x = (x.clamp(-1, 1) + 1) * 0.5
    grid = make_grid(x, nrow=int(n**0.5))
    save_image(grid, out_path)
    print(f"✅ saved to {out_path}")

if __name__ == "__main__":
    sample_rectified_flow()
