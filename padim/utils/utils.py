import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def compute_auroc(anomaly_map: NDArray, mask: NDArray) -> float:

    num_data = len(anomaly_map)
    y_score = anomaly_map.reshape(num_data, -1).max(axis=1)  # (num_data,)
    y_true = mask.reshape(num_data, -1).max(axis=1)  # (num_data,)

    score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {round(score, 3)}")
    plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
    plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()
    return score


def savegif(category: str, imgs: NDArray, masks: NDArray, amaps: NDArray) -> None:

    for i, (img, mask, amap) in enumerate(zip(imgs, masks, amaps)):

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(12, 4)),
            rect=111,
            nrows_ncols=(1, 3),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = denormalize(img)

        grid[0].imshow(img)
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("Input Image", fontsize=14)

        grid[1].imshow(img)
        grid[1].imshow(mask, alpha=0.3, cmap="Reds")
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Ground Truth", fontsize=14)

        grid[2].imshow(img)
        im = grid[2].imshow(amap, alpha=0.3, cmap="jet", vmin=0, vmax=1)
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.toggle_label(True)
        grid[2].set_title("Anomaly Map", fontsize=14)

        plt.colorbar(im, cax=grid.cbar_axes[0])
        plt.savefig(f"{category}_{i}.png", bbox_inches="tight")
        plt.close()

    # NOTE(inoue): The gif files converted by PIL or imageio were low-quality.
    #              So, I used the conversion command (ImageMagick) instead.
    subprocess.run(f"convert -delay 100 -loop 0 {category}_*.png {category}.gif", shell=True)
    subprocess.run(f"rm {category}_*.png", shell=True)


def denormalize(img: NDArray) -> NDArray:

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255.0
    return img.astype(np.uint8)
