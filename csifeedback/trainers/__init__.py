"""训练器工厂。"""

import torch

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.trainers.clnet import CLNetTrainer
from csifeedback.trainers.csinet import CsiNetTrainer
from csifeedback.trainers.stnet import STNetTrainer
from csifeedback.utils.config import ExperimentConfig


__all__ = ["BaseTrainer", "get_trainer", "CLNetTrainer", "STNetTrainer", "CsiNetTrainer"]


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
    """根据配置返回对应的训练器。"""
    name = config.model.name.lower()
    if name not in _TRAINER_MAP:
        raise ValueError(
            f"训练器选择中遇到未知模型：{name}。"
            f"请在 csifeedback/trainers/__init__.py 中注册对应的训练器。"
        )
    return _TRAINER_MAP[name](config, model, device)
