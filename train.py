# train.py
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.cifar10 import cifar10_loaders
from models.unet import TinyUNet
from utils.schedules import make_beta_schedule_cosine as make_beta_schedule, compute_alphas
from methods.ddpm import training_loss
from utils.ema import EMA

def main():
    # Hyperparameters
    batch_size = 256
    num_workers = 0        
    lr = 2e-4
    epochs = 100
    T = 1000               # Number of diffusion steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Data loaders
    train_loader, _ = cifar10_loaders(batch_size, num_workers=num_workers)

    # Model & optimizer
    model = TinyUNet(base=128, time_dim=128, out_channels=3).to(device)
    ema = EMA(model, decay=0.999)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Beta schedule and precompute alphas
    betas = make_beta_schedule(T).to(device)
    _, alpha_bars = compute_alphas(betas)

    # Training loop
    model.train()
    os.makedirs("runs/ddpm", exist_ok=True)
    global_step = 0

    for ep in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {ep+1}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)  # [-1,1]

            # random time step for each image in the batch
            t_int = torch.randint(0, T, (x0.size(0),), device=device, dtype=torch.long)

            loss = training_loss(model, x0, t_int, alpha_bars)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # update EMA every step
            ema.update(model)

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save model checkpoint (both raw and EMA)
        torch.save(model.state_dict(), "runs/ddpm/latest.pt")
        torch.save(ema.state_dict(),   "runs/ddpm/latest_ema.pt")

    print("✅ Done. Model saved at runs/ddpm/latest.pt and runs/ddpm/latest_ema.pt")

if __name__ == "__main__":
    main()