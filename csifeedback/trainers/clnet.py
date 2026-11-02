"""CLNet 训练器：单 Adam 优化器，可选 cosine warmup 学习率调度。"""

from typing import Any, Dict, Optional

import torch
from torch.optim import Adam

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.utils.config import ExperimentConfig
from csifeedback.utils.scheduler import (
    FakeLR,
    WarmUpCosineAnnealingLR,
    configure_warmup_cosine,
)


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

    def _prepare_scheduler(self, train_loader) -> None:
        if not self._lazy_scheduler or not isinstance(
            self.scheduler, WarmUpCosineAnnealingLR
        ):
            return
        T_max, T_warmup = configure_warmup_cosine(
            self.scheduler,
            total_epochs=self.training_config.epochs,
            steps_per_epoch=len(train_loader),
            T_max=self.config.scheduler.T_max,
            T_warmup=self.config.scheduler.T_warmup,
        )
        self.logger.info(
            "CLNet cosine 调度器：T_max=%d 步，T_warmup=%d 步",
            T_max,
            T_warmup,
        )
