"""CLNet 训练器：单 Adam 优化器，可选 cosine warmup 学习率调度。"""

from typing import Any, Dict, Optional

import torch
from torch.optim import Adam

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.utils.config import ExperimentConfig
from csifeedback.utils.scheduler import FakeLR, WarmUpCosineAnnealingLR


__all__ = ["CLNetTrainer"]


class CLNetTrainer(BaseTrainer):
    """CLNet 训练器。

    使用单个 Adam 优化器。当 ``config.scheduler.name == "cosine"`` 时启用 cosine warmup 调度器。
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: CSIAutoencoder,
        device: torch.device,
    ):
        optimizer = Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )

        scheduler: Optional[Any] = None
        if config.scheduler.name == "cosine":
            # 总步数与 warmup 步数在 fit() 中根据加载器长度延迟计算。
            scheduler = WarmUpCosineAnnealingLR(
                optimizer,
                T_max=config.scheduler.T_max or 1,
                T_warmup=config.scheduler.T_warmup or 0,
                eta_min=config.scheduler.eta_min,
            )
            self._lazy_scheduler = True
        else:
            scheduler = FakeLR(optimizer)
            self._lazy_scheduler = False

        super().__init__(config, model, device, optimizer, scheduler)

    def fit(self, train_loader, val_loader=None, test_loader=None):
        if self._lazy_scheduler and isinstance(self.scheduler, WarmUpCosineAnnealingLR):
            steps_per_epoch = len(train_loader)
            total_epochs = self.training_config.epochs
            T_max = self.config.scheduler.T_max or (total_epochs * steps_per_epoch)
            T_warmup = self.config.scheduler.T_warmup or (30 * steps_per_epoch)
            # warmup 步数不能超过总调度步数。
            T_warmup = min(T_warmup, max(0, T_max - 1))
            self.scheduler.T_max = T_max
            self.scheduler.T_warmup = T_warmup
            self.logger.info(
                "CLNet cosine 调度器：T_max=%d 步，T_warmup=%d 步",
                T_max,
                T_warmup,
            )
        return super().fit(train_loader, val_loader, test_loader)
