import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray


def savefig(imgs: NDArray, masks: NDArray, amaps: NDArray) -> None:

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
        im = grid[2].imshow(amap, alpha=0.3, cmap="jet")
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.colorbar(im)
        grid[2].cax.toggle_label(True)
        grid[2].set_title("Anomaly Map", fontsize=14)

        plt.savefig(f"{i}.png", bbox_inches="tight")
        plt.close()


def denormalize(img: NDArray) -> NDArray:

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255.0
    return img.astype(np.uint8)
