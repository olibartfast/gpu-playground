#!/usr/bin/env python3
"""
Train the deep-learning-inference network on synthetic RGBA→grayscale data
and export the resulting weights to weights.bin.

Usage:
    python generate_weights.py [--epochs N] [--out weights.bin]

The network is trained with a toy task: reproduce the luminance of random
RGBA images.  The resulting weights are not semantically meaningful but are
statistically similar (same order of magnitude) to a real checkpoint, which
is all the CUDA exercise needs.

Requirements:
    pip install torch numpy
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running from any directory by adding this file's directory to sys.path.
sys.path.insert(0, os.path.dirname(__file__))
from network_definition import Network


def make_batch(batch_size: int, height: int = 256, width: int = 256, device="cpu"):
    """
    Synthetic training pair: random RGBA float32 input + its weighted-average
    grayscale as target (BT.601 luminance of the RGB channels, ignoring alpha).
    """
    rgba   = torch.rand(batch_size, 4, height, width, device=device)
    # Luminance target in [0, 1], shape (N, 1, H, W)
    luma   = (0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]).unsqueeze(1)
    return rgba, luma


def train(epochs: int, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    net = Network().to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    # Use small images during training to keep it fast
    H, W = 128, 128

    print(f"Training for {epochs} epoch(s) on synthetic data…")
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        n_batches = 10  # 10 random batches per epoch

        for _ in range(n_batches):
            rgba, target = make_batch(4, H, W, device)
            opt.zero_grad()
            pred = net(rgba)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / n_batches
        print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.6f}")

    net.eval()
    net.export_weights(output_path)
    print(f"Weights saved to: {output_path}")

    # Quick sanity check: forward pass with a dummy image
    with torch.no_grad():
        dummy = torch.rand(1, 4, 128, 128, device=device)
        out   = net(dummy)
    print(f"Forward pass OK — output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weights.bin for the deep_learning_inference CUDA exercise")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "weights.bin"),
                        help="Output path for weights.bin")
    args = parser.parse_args()

    train(args.epochs, args.out)
