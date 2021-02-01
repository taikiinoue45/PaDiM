from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from torchvision.models import wide_resnet50_2


class WideResNet50(Module):
    def __init__(self) -> None:

        super().__init__()
        self.wide_resnet50 = wide_resnet50_2(pretrained=True)
        self.wide_resnet50.layer1[-1].register_forward_hook(self.hook)
        self.wide_resnet50.layer2[-1].register_forward_hook(self.hook)
        self.wide_resnet50.layer3[-1].register_forward_hook(self.hook)
        self.features: List[Tensor] = []

    def hook(self, module: Module, x: Tensor, y: Tensor) -> None:

        self.features.append(y.cpu().detach())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        self.wide_resnet50(x)
        feature1, feature2, feature3 = self.features
        self.features = []
        return (feature1, feature2, feature3)
