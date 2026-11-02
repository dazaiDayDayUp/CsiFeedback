"""训练器工厂。

baseline 模型使用各自的专用 trainer 以保持原始训练行为；未注册的新模型
自动回退到配置驱动的 :class:`DefaultTrainer`。
"""

import torch

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.trainers.clnet import CLNetTrainer
from csifeedback.trainers.csinet import CsiNetTrainer
from csifeedback.trainers.default import DefaultTrainer
from csifeedback.trainers.stnet import STNetTrainer
from csifeedback.utils.config import ExperimentConfig
from csifeedback.utils.logging import get_logger


__all__ = [
    "BaseTrainer",
    "DefaultTrainer",
    "get_trainer",
    "CLNetTrainer",
    "STNetTrainer",
    "CsiNetTrainer",
]


_TRAINER_MAP = {
    "clnet": CLNetTrainer,
    "stnet": STNetTrainer,
    "csinet": CsiNetTrainer,
}


def get_trainer(
    config: ExperimentConfig,
    model: CSIAutoencoder,
    device: torch.device,
) -> BaseTrainer:
    """根据配置返回对应的训练器。未注册的模型使用 ``DefaultTrainer``。"""
    name = config.model.name.lower()
    trainer_cls = _TRAINER_MAP.get(name)
    if trainer_cls is None:
        get_logger(__name__).info(
            "模型 %s 没有专用 trainer，使用 DefaultTrainer（配置驱动）", name
        )
        return DefaultTrainer(config, model, device)
    return trainer_cls(config, model, device)
