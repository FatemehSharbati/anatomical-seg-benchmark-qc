"""Overlay QC: prediction contours over the image, mid-slices -> PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_overlay(image, pred, target, out_png, n_slices=5):
    """Green = prediction contour, red = reference contour, on the image."""
    image = np.asarray(image, dtype=np.float32)
    pred = np.asarray(pred)
    target = np.asarray(target)
    d = image.shape[2]
    idx = np.linspace(int(d * 0.2), int(d * 0.8), n_slices).astype(int)
    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    axes = np.atleast_1d(axes).ravel()
    vmin, vmax = np.percentile(image, [1, 99])
    for ax, z in zip(axes, idx):
        ax.imshow(np.rot90(image[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(np.rot90((pred[:, :, z] > 0).astype(float)), colors="lime",
                   linewidths=0.8)
        ax.contour(np.rot90((target[:, :, z] > 0).astype(float)), colors="red",
                   linewidths=0.8, linestyles="dashed")
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png
