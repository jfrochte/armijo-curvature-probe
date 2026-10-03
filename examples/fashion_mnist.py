"""Fashion-MNIST: Adam with a far too large learning rate, with and without the probe.

    python examples/fashion_mnist.py                 # lr = 1.0, probe on and off
    python examples/fashion_mnist.py --lr 1e-3      # cap does not bind: identical runs

Downloads Fashion-MNIST via torchvision into ./data on first use.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armijo_probe import cap_learning_rate  # noqa: E402


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
    )


def loaders(batch_size, seed):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
            torch.utils.data.DataLoader(test, batch_size=1000))


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        correct += (model(X).argmax(1) == Y).sum().item()
        total += Y.numel()
    model.train()
    return correct / total


def run(lr, use_probe, epochs, seed, device):
    torch.manual_seed(seed)
    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = loaders(128, seed)

    first_loss = None
    for epoch in range(epochs):
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            if use_probe and first_loss is None:
                # One Armijo probe on the first training mini-batch, before the first step.
                res = cap_learning_rate(optimizer, model, lambda: criterion(model(X), Y), kappa=2.0)
                print(f"  probe: alpha = {res.alpha:.3g} (accepted: {res.accepted}), "
                      f"lr {lr:g} -> {optimizer.param_groups[0]['lr']:.3g}")
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
            first_loss = first_loss or loss.item()
            if not torch.isfinite(loss) or loss.item() > 5 * first_loss:
                print(f"  diverged in epoch {epoch + 1} (loss {loss.item():.3g})")
                return
        print(f"  epoch {epoch + 1}: test accuracy {accuracy(model, test_loader, device):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True  # makes the two runs comparable bit for bit
    torch.backends.cudnn.benchmark = False
    for use_probe in (False, True):
        print(f"Adam, lr = {args.lr:g}, {'with' if use_probe else 'without'} probe")
        run(args.lr, use_probe, args.epochs, args.seed, device)


if __name__ == "__main__":
    main()
