from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray as NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from torch import Tensor

from padim.metrics import compute_auroc
from padim.runner import BaseRunner
from padim.utils import savefig


class Runner(BaseRunner):
    def _train(self) -> Tuple[NDArray, NDArray]:

        self.model.eval()
        features: Dict[str, List[Tensor]] = {"feature1": [], "feature2": [], "feature3": []}
        for _, mb_img, _ in self.dataloaders["train"]:

            mb_img = mb_img.to(self.cfg.params.device)
            with torch.no_grad():
                feature1, feature2, feature3 = self.model(mb_img)
            features["feature1"].append(feature1)
            features["feature2"].append(feature2)
            features["feature3"].append(feature3)

        embeddings = self._embed(features)
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w)

        means = embeddings.mean(axis=0)
        covariances = np.zeros((c, c, h * w))
        identity = np.identity(c)
        for i in range(h * w):
            covariances[:, :, i] = np.cov(embeddings[:, :, i], rowvar=False) + 0.01 * identity

        return (means, covariances)

    def _test(self, means: NDArray, covariances: NDArray) -> None:

        self.model.eval()
        features: Dict[str, List[Tensor]] = {"feature1": [], "feature2": [], "feature3": []}
        artifacts: Dict[str, List[NDArray]] = {"img": [], "mask": []}
        for _, mb_img, mb_mask in self.dataloaders["test"]:

            mb_img = mb_img.to(self.cfg.params.device)
            with torch.no_grad():
                feature1, feature2, feature3 = self.model(mb_img)
            features["feature1"].append(feature1)
            features["feature2"].append(feature2)
            features["feature3"].append(feature3)

            artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).cpu().detach().numpy())
            artifacts["mask"].extend(mb_mask.cpu().detach().numpy())

        embeddings = self._embed(features)
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w)

        distances = []
        for i in range(h * w):
            mean = means[:, i]
            covariance_inv = np.linalg.inv(covariances[:, :, i])
            distance = [mahalanobis(e[:, i], mean, covariance_inv) for e in embeddings]
            distances.append(distance)

        distance_map = np.array(distances).transpose(1, 0).reshape(b, h, w)
        distance_map = torch.tensor(distance_map)  # (b, h, w)
        distance_map = distance_map.unsqueeze(1)  # (b, 1, h, w)

        anomaly_map = (
            F.interpolate(
                distance_map,
                size=mb_img.shape[2],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        for i in range(len(anomaly_map)):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        auroc = compute_auroc(anomaly_map, np.array(artifacts["mask"]))
        mlflow.log_metric("AUROC", auroc)
        savefig(np.array(artifacts["img"]), np.array(artifacts["mask"]), anomaly_map)

        print("image ROCAUC: %.3f" % (auroc))

    def _embed(self, features: Dict[str, List[Tensor]]) -> NDArray:

        embeddings = torch.cat(features["feature1"], dim=0)
        embeddings = self._embeddings_concat(embeddings, torch.cat(features["feature2"], dim=0))
        embeddings = self._embeddings_concat(embeddings, torch.cat(features["feature3"], dim=0))
        embeddings = torch.index_select(embeddings, dim=1, index=self.embedding_ids)
        return embeddings.numpy()

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
