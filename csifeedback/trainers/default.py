"""通用训练器：完全由配置驱动，供没有特殊训练逻辑的新模型直接复用。"""

from typing import Any

import torch
from torch.optim import Adam, AdamW, Optimizer, SGD

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.utils.config import ExperimentConfig
from csifeedback.utils.scheduler import (
    FakeLR,
    WarmUpCosineAnnealingLR,
    configure_warmup_cosine,
)


__all__ = ["DefaultTrainer", "build_optimizer"]


def build_optimizer(model: CSIAutoencoder, config: ExperimentConfig) -> Optimizer:
    """根据 ``config.optimizer`` 构建优化器。"""
    name = config.optimizer.name.lower().strip()
    kwargs: dict[str, Any] = {
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
    }

    if name == "adam":
        return Adam(model.parameters(), betas=config.optimizer.betas, **kwargs)
    if name == "adamw":
        return AdamW(model.parameters(), betas=config.optimizer.betas, **kwargs)
    if name == "sgd":
        return SGD(model.parameters(), momentum=config.optimizer.momentum, **kwargs)

    raise ValueError(
        f"未知优化器：{name}。可选：adam、adamw、sgd；"
        "如需特殊优化器，请实现专用 trainer。"
    )


class DefaultTrainer(BaseTrainer):
    """配置驱动的通用训练器。

    - 优化器：adam / adamw / sgd（见 :func:`build_optimizer`）
    - 调度器：``const`` 或 ``cosine``（warmup + 余弦退火）
    - 最优模型：由 ``training.select_best`` 决定；默认 ``all``
      （同时保存 test rho / test NMSE 最优，与 CLNet、STNet 一致）

    新模型如果没有自定义 loss、双优化器等需求，注册模型后即可直接使用本训练器，
    不需要再修改 ``csifeedback/trainers/__init__.py``。
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: CSIAutoencoder,
        device: torch.device,
    ):
        optimizer = build_optimizer(model, config)

        scheduler_name = config.scheduler.name.lower().strip()
        if scheduler_name == "cosine":
            scheduler: Any = WarmUpCosineAnnealingLR(
                optimizer,
                T_max=config.scheduler.T_max or 1,
                T_warmup=config.scheduler.T_warmup or 0,
                eta_min=config.scheduler.eta_min,
            )
            self._lazy_scheduler = True
        elif scheduler_name in {"const", "none", ""}:
            scheduler = FakeLR(optimizer)
            self._lazy_scheduler = False
        else:
            raise ValueError(
                f"未知调度器：{scheduler_name}。可选：const、cosine；"
                "如需特殊调度器，请实现专用 trainer。"
            )

        super().__init__(config, model, device, optimizer, scheduler)

    def _prepare_scheduler(self, train_loader) -> None:
        if not self._lazy_scheduler:
            return
        T_max, T_warmup = configure_warmup_cosine(
            self.scheduler,
            total_epochs=self.training_config.epochs,
            steps_per_epoch=len(train_loader),
            T_max=self.config.scheduler.T_max,
            T_warmup=self.config.scheduler.T_warmup,
        )
        self.logger.info(
            "cosine 调度器：T_max=%d 步，T_warmup=%d 步",
            T_max,
            T_warmup,
        )
