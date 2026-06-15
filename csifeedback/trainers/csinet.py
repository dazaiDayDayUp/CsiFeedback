"""CsiNet 训练器：单 Adam 优化器，按验证损失保存最优模型。"""

import os
from typing import Any, Optional

import torch
from torch.optim import Adam

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.utils.config import ExperimentConfig
from csifeedback.utils.scheduler import FakeLR


__all__ = ["CsiNetTrainer"]


class CsiNetTrainer(BaseTrainer):
    """CsiNet 训练器。

    使用单个 Adam 优化器，并按照验证损失保存最优模型，与原始 PyTorch 复现一致。
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
        scheduler = FakeLR(optimizer)
        super().__init__(config, model, device, optimizer, scheduler)

    def _maybe_save_best(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint("best_val.pth")
            self.logger.info(
                "新的最优验证损失：%.4e（已保存 best_val.pth）",
                val_loss,
            )
