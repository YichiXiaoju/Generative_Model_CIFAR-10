import os
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.cifar10 import cifar10_loaders
from models.unet import TinyUNet
from methods.flowmatching import training_loss_flowmatching, sample_xt
from utils.ema import EMA

def main():
    batch_size = 128
    num_workers = 4
    epochs = 50
    lr = 2e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _ = cifar10_loaders(batch_size=batch_size, num_workers=num_workers)
    net = TinyUNet(base=64, time_dim=128, out_channels=3).to(device)
    ema = EMA(net, beta=0.999)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs("runs/flowmatching", exist_ok=True)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x0,_ in pbar:
            x0 = x0.to(device)  # (n,3,32,32), scaled to [-1,1]
            loss = training_loss_flowmatching(net, x0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            ema.update()

            pbar.set_postfix({"loss": loss.item()})
        torch.save(net.state_dict(), "runs/flowmatching/latest.pt")
        torch.save(ema.state_dict(), "runs/flowmatching/latest_ema.pt")

    print("✅ Flow Matching training finished.")


if __name__ == "__main__":
    main()