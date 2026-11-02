"""训练器使用的学习率调度器。"""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ["FakeLR", "WarmUpCosineAnnealingLR", "configure_warmup_cosine"]


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
            # warmup 阶段：学习率从 0 线性增长到初始学习率，与 CLNet baseline 协议一致。
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


def configure_warmup_cosine(
    scheduler: WarmUpCosineAnnealingLR,
    total_epochs: int,
    steps_per_epoch: int,
    T_max: int | None = None,
    T_warmup: int | None = None,
) -> tuple[int, int]:
    """根据 epoch 数与 dataloader 长度计算 cosine 调度的总步数和 warmup 步数。

    ``T_max`` / ``T_warmup`` 为 ``None`` 时分别取“全部训练步数”和“30 个 epoch”。
    warmup 步数会被裁剪到不超过总步数减一。
    """
    resolved_T_max = T_max or (total_epochs * steps_per_epoch)
    resolved_T_warmup = T_warmup or (30 * steps_per_epoch)

    if resolved_T_max < 1:
        raise ValueError("cosine 调度的总步数必须大于 0")
    resolved_T_warmup = min(resolved_T_warmup, max(0, resolved_T_max - 1))

    scheduler.T_max = resolved_T_max
    scheduler.T_warmup = resolved_T_warmup
    return resolved_T_max, resolved_T_warmup
