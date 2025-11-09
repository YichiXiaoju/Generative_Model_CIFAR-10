# train_rectified_flow.py
import os, torch
import torch.optim as optim
from tqdm import tqdm

from datasets.cifar10 import cifar10_loaders
from models.unet import TinyUNet
from methods.rectified_flow import training_loss_rectified_flow
from utils.ema import EMA

def main():
    batch_size = 128
    num_workers = 8
    lr = 2e-4
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, _ = cifar10_loaders(batch_size, num_workers=num_workers)

    model = TinyUNet(base=128, time_dim=128, out_channels=3).to(device)
    ema = EMA(model, decay=0.999)
    optimz = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs("runs/rectified_flow", exist_ok=True)

    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"[RF] Epoch {ep+1}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)
            loss = training_loss_rectified_flow(model, x0)

            optimz.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimz.step()
            ema.update(model)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save(model.state_dict(), "runs/rectified_flow/latest.pt")
        torch.save(ema.state_dict(), "runs/rectified_flow/latest_ema.pt")

    print("✅ Rectified Flow training finished. Check runs/rectified_flow/")

if __name__ == "__main__":
    main()
