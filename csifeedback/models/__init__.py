"""CSI 反馈自编码器的模型工厂。"""

from typing import Any, Dict

import torch.nn as nn

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.clnet import create_clnet
from csifeedback.models.csinet import create_csinet
from csifeedback.models.stnet import create_stnet


__all__ = ["CSIAutoencoder", "get_model", "create_clnet", "create_stnet", "create_csinet"]


_MODEL_FACTORIES = {
    "clnet": create_clnet,
    "stnet": create_stnet,
    "csinet": create_csinet,
}


def get_model(name: str, **kwargs) -> CSIAutoencoder:
    """按名称创建模型。

    参数：
        name: ``clnet``、``stnet`` 或 ``csinet``。
        **kwargs: 模型特定参数。
            - CLNet: ``reduction`` (int)
            - STNet/CsiNet: ``encoded_dim`` (int)

    返回：
        :class:`CSIAutoencoder` 的实例。
    """
    name = name.lower().strip()
    if name not in _MODEL_FACTORIES:
        raise ValueError(f"未知模型：{name}。可选：{list(_MODEL_FACTORIES.keys())}")
    model = _MODEL_FACTORIES[name](**kwargs)
    return model


def get_model_from_config(model_config: Any) -> CSIAutoencoder:
    """从 ModelConfig dataclass 创建模型，支持 extra 参数字段。"""
    name = model_config.name
    factory = _MODEL_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"未知模型：{name}。请在 csifeedback/models/__init__.py 中注册对应的工厂函数。"
        )

    kwargs: Dict[str, Any] = {}
    if model_config.reduction is not None:
        kwargs["reduction"] = model_config.reduction
    if model_config.encoded_dim is not None:
        kwargs["encoded_dim"] = model_config.encoded_dim
    kwargs.update(model_config.extra)
    return factory(**kwargs)
