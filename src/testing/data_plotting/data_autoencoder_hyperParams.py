import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_PATH = "loss_contrastive.json"
OUT_DIR = Path("contrastive_plots")
OUT_DIR.mkdir(exist_ok=True)

with open(FILE_PATH, "r") as f:
    data = json.load(f)

lambdas = [0.01 * i for i in range(len(data))]

metrics = ["total", "mse", "ce"]

# Individual plots for each lambda
for run_idx, run in enumerate(data):
    lam = lambdas[run_idx]
    epochs = np.arange(1, len(run) + 1)

    for metric in metrics:
        values = [epoch_data[metric] for epoch_data in run]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values)
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper() + " Loss")
        plt.title(f"{metric.upper()} Loss vs Epoch | lambda = {lam:.2f}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{metric}_lambda_{lam:.2f}.png", dpi=300)
        plt.show()

# Combined comparison plots
for metric in metrics:
    plt.figure(figsize=(9, 6))

    for run_idx, run in enumerate(data):
        lam = lambdas[run_idx]
        epochs = np.arange(1, len(run) + 1)
        values = [epoch_data[metric] for epoch_data in run]
        plt.plot(epochs, values, label=f"lambda={lam:.2f}")

    plt.xlabel("Epoch")
    plt.ylabel(metric.upper() + " Loss")
    plt.title(f"{metric.upper()} Loss vs Epoch for Different Lambdas")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"combined_{metric}.png", dpi=300)
    plt.show()

# 3D plot: loss, epoch, lambda
for metric in metrics:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for run_idx, run in enumerate(data):
        lam = lambdas[run_idx]
        epochs = np.arange(1, len(run) + 1)
        lambda_values = np.full_like(epochs, lam, dtype=float)
        losses = np.array([epoch_data[metric] for epoch_data in run])

        ax.plot(epochs, lambda_values, losses, label=f"lambda={lam:.2f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lambda")
    ax.set_zlabel(metric.upper() + " Loss")
    ax.set_title(f"3D {metric.upper()} Loss Surface")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"3d_{metric}_loss.png", dpi=300)
    plt.show()
    