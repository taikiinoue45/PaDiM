from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray as NDArray
from scipy.spatial.distance import mahalanobis
from torch import Tensor
from tqdm import tqdm
from typing_extensions import Literal

from padim.runner import BaseRunner
from padim.utils import compute_auroc, mean_smoothing, savegif


class Runner(BaseRunner):
    def _train(self) -> Tuple[NDArray, NDArray]:

        embeddings, _ = self._embed("train")
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w)

        means = embeddings.mean(axis=0)
        cvars = np.zeros((c, c, h * w))
        identity = np.identity(c)
        for i in range(h * w):
            cvars[:, :, i] = np.cov(embeddings[:, :, i], rowvar=False) + 0.01 * identity

        return (means, cvars)

    def _test(self, means: NDArray, cvars: NDArray) -> None:

        embeddings, artifacts = self._embed("test")
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w)

        distances = []
        for i in range(h * w):
            mean = means[:, i]
            cvar_inv = np.linalg.inv(cvars[:, :, i])
            distance = [mahalanobis(e[:, i], mean, cvar_inv) for e in embeddings]
            distances.append(distance)

        img_h = self.cfg.params.height
        img_w = self.cfg.params.width
        amaps = torch.tensor(np.array(distances), dtype=torch.float32)
        amaps = amaps.permute(1, 0).view(b, h, w).unsqueeze(dim=1)  # (b, 1, h, w)
        amaps = F.interpolate(amaps, size=(img_h, img_w), mode="bilinear", align_corners=False)
        amaps = mean_smoothing(amaps)
        amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())
        amaps = amaps.squeeze().numpy()

        auroc = compute_auroc(amaps, np.array(artifacts["mask"]))
        mlflow.log_metric("AUROC", auroc)
        savegif(
            self.cfg.params.category,
            np.array(artifacts["image"]),
            np.array(artifacts["mask"]),
            amaps,
        )

    def _embed(self, mode: Literal["train", "test"]) -> Tuple[NDArray, Dict[str, List[NDArray]]]:

        self.model.eval()
        features: Dict[str, List[Tensor]] = {"feature1": [], "feature2": [], "feature3": []}
        artifacts: Dict[str, List[NDArray]] = {"image": [], "mask": []}
        pbar = tqdm(self.dataloaders[mode], desc=f"{self.cfg.params.category} - {mode}")
        for _, imgs, masks in pbar:

            with torch.no_grad():
                feature1, feature2, feature3 = self.model(imgs.to(self.cfg.params.device))
            features["feature1"].append(feature1)
            features["feature2"].append(feature2)
            features["feature3"].append(feature3)
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
