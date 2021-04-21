from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose
from mvtec.builder import Builder
from mvtec.metrics import compute_pro, compute_roc
from mvtec.utils import savegif
from numpy import ndarray
from omegaconf.dictconfig import DictConfig
from scipy.spatial.distance import mahalanobis
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import Literal


class Runner(Builder):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()

        self.params: DictConfig = cfg.params
        self.transform: Dict[str, Compose] = {
            key: Compose(self.build_list_cfg(cfg)) for key, cfg in cfg.transform.items()
        }
        self.dataset: Dict[str, Dataset] = {
            key: self.build_dict_cfg(cfg, transform=self.transform[key])
            for key, cfg in cfg.dataset.items()
        }
        self.dataloader: Dict[str, DataLoader] = {
            key: self.build_dict_cfg(cfg, dataset=self.dataset[key])
            for key, cfg in cfg.dataloader.items()
        }
        self.model: Module = self.build_dict_cfg(cfg.model).to(self.params.device)
        self.embedding_ids = torch.randperm(1792)[: self.params.num_embedding]

    def run(self) -> None:

        mean, covariance = self._train()
        self._test(mean, covariance)
        torch.save(self.model.state_dict(), "model.pth")

    def _train(self) -> Tuple[ndarray, ndarray]:

        embeddings, _ = self._embed("train")
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w)

        means = embeddings.mean(axis=0)
        cvars = np.zeros((c, c, h * w))
        identity = np.identity(c)
        for i in tqdm(range(h * w), desc=f"{self.params.category} - compute covariance"):
            cvars[:, :, i] = np.cov(embeddings[:, :, i], rowvar=False) + 0.01 * identity

        return (means, cvars)

    def _test(self, means: ndarray, cvars: ndarray) -> None:

        embeddings, artifacts = self._embed("test")
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w)

        distances = []
        for i in tqdm(range(h * w), desc=f"{self.params.category} - compute distance"):
            mean = means[:, i]
            cvar_inv = np.linalg.inv(cvars[:, :, i])
            distance = [mahalanobis(e[:, i], mean, cvar_inv) for e in embeddings]
            distances.append(distance)

        img_h = self.params.height
        img_w = self.params.width
        amaps = torch.tensor(np.array(distances), dtype=torch.float32)
        amaps = amaps.permute(1, 0).view(b, h, w).unsqueeze(dim=1)  # (b, 1, h, w)
        amaps = F.interpolate(amaps, size=(img_h, img_w), mode="bilinear", align_corners=False)
        amaps = self._mean_smoothing(amaps)
        amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())
        amaps = amaps.squeeze().numpy()

        imgs = self._denormalize(np.array(artifacts["image"]))
        masks = np.array(artifacts["mask"])

        num_data = len(amaps)
        y_trues = masks.reshape(num_data, -1).max(axis=1)
        y_preds = amaps.reshape(num_data, -1).max(axis=1)

        compute_roc(y_trues, y_preds, artifacts["stem"])
        compute_pro(masks, amaps)
        savegif(imgs, masks, amaps)

    def _embed(self, mode: Literal["train", "test"]) -> Tuple[ndarray, Dict[str, List[ndarray]]]:

        self.model.eval()
        features: Dict[str, List[Tensor]] = {"feature1": [], "feature2": [], "feature3": []}
        artifacts: Dict[str, List[Union[str, ndarray]]] = {"stem": [], "image": [], "mask": []}
        pbar = tqdm(self.dataloader[mode], desc=f"{self.params.category} - {mode}")
        for stems, imgs, masks in pbar:

            with torch.no_grad():
                feature1, feature2, feature3 = self.model(imgs.to(self.params.device))
            features["feature1"].append(feature1)
            features["feature2"].append(feature2)
            features["feature3"].append(feature3)
            artifacts["stem"].extend(stems)
            artifacts["image"].extend(imgs.permute(0, 2, 3, 1).cpu().detach().numpy())
            artifacts["mask"].extend(masks.cpu().detach().numpy())

        embeddings = torch.cat(features["feature1"], dim=0)
        embeddings = self._embeddings_concat(embeddings, torch.cat(features["feature2"], dim=0))
        embeddings = self._embeddings_concat(embeddings, torch.cat(features["feature3"], dim=0))
        embeddings = torch.index_select(embeddings, dim=1, index=self.embedding_ids)
        return (embeddings.numpy(), artifacts)

    def _embeddings_concat(self, x0: Tensor, x1: Tensor) -> Tensor:

        b0, c0, h0, w0 = x0.size()
        b1, c1, h1, w1 = x1.size()
        s = h0 // h1
        x0 = F.unfold(x0, kernel_size=(s, s), dilation=(1, 1), stride=(s, s))
        x0 = x0.view(b0, c0, -1, h1, w1)
        z = torch.zeros(b0, c0 + c1, x0.size(2), h1, w1)
        for i in range(x0.size(2)):
            z[:, :, i, :, :] = torch.cat((x0[:, :, i, :, :], x1), 1)
        z = z.view(b0, -1, h1 * w1)
        z = F.fold(z, kernel_size=(s, s), output_size=(h0, w0), stride=(s, s))
        return z

    def _mean_smoothing(self, amaps: Tensor, kernel_size: int = 21) -> Tensor:

        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
        mean_kernel = mean_kernel.to(amaps.device)
        return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)

    def _denormalize(self, imgs: ndarray) -> ndarray:

        mean = np.array(self.params.normalize_mean)
        std = np.array(self.params.normalize_std)
        imgs = (imgs * std + mean) * 255.0
        return imgs.astype(np.uint8)
