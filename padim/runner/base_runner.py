from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Tuple

import torch
from albumentations import Compose
from numpy import ndarray as NDArray
from omegaconf.dictconfig import DictConfig
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class BaseRunner(ABC):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()
        self.cfg = cfg
        self.transforms = {k: self._init_transforms(k) for k in self.cfg.transforms.keys()}
        self.datasets = {k: self._init_datasets(k) for k in self.cfg.datasets.keys()}
        self.dataloaders = {k: self._init_dataloaders(k) for k in self.cfg.dataloaders.keys()}
        self.model = self._init_model().to(self.cfg.params.device)

        self.embedding_ids = torch.randperm(1792)[: self.cfg.params.num_embedding]

    def _init_transforms(self, key: str) -> Compose:

        transforms = []
        for cfg in self.cfg.transforms[key]:
            attr = self._get_attr(cfg.name)
            transforms.append(attr(**cfg.get("args", {})))
        return Compose(transforms)

    def _init_datasets(self, key: str) -> Dataset:

        cfg = self.cfg.datasets[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), mode=key, transforms=self.transforms[key])

    def _init_dataloaders(self, key: str) -> DataLoader:

        cfg = self.cfg.dataloaders[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.datasets[key])

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    def run(self) -> None:

        mean, covariance = self._train()
        self._test(mean, covariance)
        torch.save(self.model.state_dict(), "model.pth")

    @abstractmethod
    def _train(self) -> Tuple[NDArray, NDArray]:

        raise NotImplementedError()

    @abstractmethod
    def _test(self, means: NDArray, covariances: NDArray) -> None:

        raise NotImplementedError()
