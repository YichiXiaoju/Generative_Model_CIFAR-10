import torch
from torchvision.utils import save_image, make_grid
from models.unet import TinyUNet
from methods.flowmatching import sample_xt

@torch.no_grad()
def sample_flowmatching(model_path, n=16, steps=50, device="cuda", output_path="samples/flowmatching.png"):
    """
    Flow matching sampling.
    net: the trained vector field network (TinyUNet)
    n: number of samples to generate
    steps: number of sampling steps
    returns: (n,3,32,32) tensor of generated samples
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TinyUNet(base=128, time_dim=128, out_channels=3).to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state, strict=True)

    x = torch.randn(n, 3, 32, 32, device=device)  # initial noise
    t = torch.linspace(1, 0, steps, device=device)

    for i in range(len(t)-1):
        dt = t[i] - t[i+1]
        v = net(x, t[i].expand(n))  # predict vector field v_theta(xt,t)
        x = x - v * dt  # Euler step

    x = (x.clamp(-1, 1) + 1) * 0.5  # scale to [0,1]
    grid = make_grid(x, nrow=4)
    save_image(grid, output_path)
    print(f"✅ saved {output_path}")