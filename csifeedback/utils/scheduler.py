"""训练器使用的学习率调度器。"""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ["FakeLR", "WarmUpCosineAnnealingLR"]


class FakeLR(_LRScheduler):
    """占位调度器，始终保持初始学习率不变。"""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class WarmUpCosineAnnealingLR(_LRScheduler):
    """先线性 warmup，再余弦退火。

    前 ``T_warmup`` 步将学习率从 ``eta_min`` 线性提升到初始学习率；
    之后进入余弦退火阶段，总共 ``T_max`` 步。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        T_warmup: int = 0,
        eta_min: float = 5e-5,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.T_warmup:
            # warmup 阶段：学习率从 0 线性增长到初始学习率，与原始 CLNet 实现一致。
            alpha = self.last_epoch / max(1, self.T_warmup)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.T_warmup) / max(
                1, self.T_max - self.T_warmup
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * progress))
                / 2
                for base_lr in self.base_lrs
            ]
