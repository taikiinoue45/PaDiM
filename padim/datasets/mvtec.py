from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from albumentations import Compose
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import Literal


class MVTecDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        query_list: List[str],
        mode: Literal["train", "test"],
        transforms: Compose,
        debug: bool,
    ) -> None:

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.debug = debug

        info_csv = pd.read_csv(self.data_dir / "info.csv")
        df = pd.concat([info_csv.query(q) for q in query_list])
        df.to_csv(f"{mode}_dataset.csv", index=False)
        mlflow.log_artifact(f"{mode}_dataset.csv")
        self.stem_list = df["stem"].tolist()

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor]:

        stem = self.stem_list[index]

        img_path = str(self.data_dir / f"images/{stem}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = str(self.data_dir / f"masks/{stem}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1

        data_dict = self.transforms(image=img, mask=mask)

        if self.debug:
            self._save_transformed_images(index, data_dict["image"], data_dict["mask"])

        return (stem, data_dict["image"], data_dict["mask"])

    def _save_transformed_images(self, index: int, img: Tensor, mask: Tensor) -> None:

        img = img.permute(1, 2, 0).detach().numpy()
        mask = mask.detach().numpy()
        plt.figure(figsize=(9, 3))

        plt.subplot(131)
        plt.title("Input Image")
        plt.imshow(img)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.title("Ground Truth")
        plt.imshow(mask)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.title("Supervision")
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.tight_layout()
        plt.savefig(f"{self.stem_list[index]}.png")

    def __len__(self) -> int:

        return len(self.stem_list)
