import os
import subprocess
from statistics import mean
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch import Tensor
from tqdm import tqdm


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def compute_roc_score(amaps: NDArray, masks: NDArray, stems: List[str]) -> float:

    num_data = len(stems)
    masks[masks != 0] = 1
    y_scores = amaps.reshape(num_data, -1).max(axis=1)
    y_trues = masks.reshape(num_data, -1).max(axis=1)
    fprs, tprs, thresholds = roc_curve(y_trues, y_scores, pos_label=1, drop_intermediate=False)

    # Save roc_curve.csv
    keys = [f"threshold_{i}" for i in range(len(thresholds))]
    roc_df = pd.DataFrame({"key": keys, "fpr": fprs, "tpr": tprs, "threshold": thresholds})
    roc_df.to_csv("roc_curve.csv", index=False)

    # Update test_dataset.csv
    pred_csv = pd.merge(
        pd.DataFrame({"stem": stems, "y_score": y_scores, "y_true": y_trues}),
        pd.read_csv("test_dataset.csv"),
        on="stem",
    )
    for i, th in enumerate(thresholds):
        pred_csv[f"threshold_{i}"] = pred_csv["y_score"].apply(lambda x: 1 if x >= th else 0)
    pred_csv.to_csv("test_dataset.csv", index=False)

    return roc_auc_score(y_trues, y_scores)


def compute_pro_score(amaps: NDArray, masks: NDArray) -> float:

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    max_step = 200
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / max_step

    for th in tqdm(np.arange(min_th, max_th, delta), desc="compute pro"):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(TP_pixels / region.area)

        inverse_masks = 1 - masks
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    df.to_csv("pro_curve.csv", index=False)
    return auc(df["fpr"], df["pro"])


def draw_roc_and_pro_curve(roc_score: float, pro_score: float) -> None:

    grid = ImageGrid(
        fig=plt.figure(figsize=(8, 8)),
        rect=111,
        nrows_ncols=(1, 1),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )

    roc_df = pd.read_csv("roc_curve.csv")
    fpr = roc_df["fpr"]
    tpr = roc_df["tpr"]
    th = roc_df["threshold"]
    v_min = th.min()
    grid[0].plot(fpr, tpr, color="k", label=f"ROC Score: {round(roc_score, 3):.3f}", zorder=1)
    im = grid[0].scatter(fpr, tpr, s=8, c=th, cmap="jet", vmin=v_min, vmax=1, zorder=2)
    grid[0].set_xlim(-0.05, 1.05)
    grid[0].set_ylim(-0.05, 1.05)
    grid[0].set_xticks(np.arange(0, 1.1, 0.1))
    grid[0].set_yticks(np.arange(0, 1.1, 0.1))
    grid[0].tick_params(axis="both", labelsize=14)
    grid[0].set_xlabel("FPR: FP / (TN + FP)", fontsize=24)
    grid[0].set_ylabel("TPR: TP / (TP + FN)", fontsize=24)
    grid[0].xaxis.set_label_coords(0.5, -0.1)
    grid[0].yaxis.set_label_coords(-0.1, 0.5)
    grid[0].legend(fontsize=24)
    grid[0].grid(which="both", linestyle="dotted", linewidth=1)
    cb = plt.colorbar(im, cax=grid.cbar_axes[0])
    cb.ax.tick_params(labelsize="large")
    plt.savefig("roc_curve.png")
    plt.close()

    grid = ImageGrid(
        fig=plt.figure(figsize=(8, 8)),
        rect=111,
        nrows_ncols=(1, 1),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )

    pro_df = pd.read_csv("pro_curve.csv")
    fpr = pro_df["fpr"]
    pro = pro_df["pro"]
    th = pro_df["threshold"]
    grid[0].plot(fpr, pro, color="k", label=f"PRO Score: {round(pro_score, 3):.3f}", zorder=1)
    im = grid[0].scatter(fpr, pro, s=8, c=th, cmap="jet", vmin=v_min, vmax=1, zorder=2)
    grid[0].set_xlim(-0.05, 1.05)
    grid[0].set_ylim(-0.05, 1.05)
    grid[0].set_xticks(np.arange(0, 1.1, 0.1))
    grid[0].set_yticks(np.arange(0, 1.1, 0.1))
    grid[0].tick_params(axis="both", labelsize=14)
    grid[0].set_xlabel("FPR: FP / (TN + FP)", fontsize=24)
    grid[0].set_ylabel("PRO: Per-Region Overlap", fontsize=24)
    grid[0].xaxis.set_label_coords(0.5, -0.1)
    grid[0].yaxis.set_label_coords(-0.1, 0.5)
    grid[0].legend(fontsize=24)
    grid[0].grid(which="both", linestyle="dotted", linewidth=1)
    cb = plt.colorbar(im, cax=grid.cbar_axes[0])
    cb.ax.tick_params(labelsize="large")
    plt.savefig("pro_curve.png")
    plt.close()


def savegif(imgs: NDArray, amaps: NDArray, masks: NDArray, stems: List[str]) -> None:

    os.mkdir("results")
    pbar = tqdm(enumerate(zip(stems, imgs, masks, amaps)), desc="savefig")
    for i, (stem, img, mask, amap) in pbar:

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
        grid[0].set_title("Input Image", fontsize=24)

        grid[1].imshow(img)
        grid[1].imshow(mask, alpha=0.3, cmap="Reds")
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Ground Truth", fontsize=24)

        grid[2].imshow(img)
        im = grid[2].imshow(amap, alpha=0.3, cmap="jet", vmin=0, vmax=1)
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.toggle_label(True)
        grid[2].set_title("Anomaly Map", fontsize=24)

        plt.colorbar(im, cax=grid.cbar_axes[0])
        plt.savefig(f"results/{stem}.png", bbox_inches="tight")
        plt.close()

    # NOTE(inoue): The gif files converted by PIL or imageio were low-quality.
    #              So, I used the conversion command (ImageMagick) instead.
    subprocess.run("convert -delay 100 -loop 0 results/*.png result.gif", shell=True)


def denormalize(img: NDArray) -> NDArray:

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255.0
    return img.astype(np.uint8)
