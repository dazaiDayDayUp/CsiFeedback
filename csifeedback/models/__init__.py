"""CSI 反馈模型的注册表与工厂。

baseline 模型（CLNet / STNet / CsiNet）在各自模块中通过 ``@register_model``
自注册。新增研究模型时：

1. 在 ``csifeedback/models/<name>.py`` 中实现网络与工厂函数；
2. 用 ``@register_model("<name>")`` 装饰工厂函数；
3. 在下方导入该模块，使其注册生效。

已有模型代码无需改动。
"""

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import (
    _MODEL_FACTORIES,
    available_models,
    get_model,
    get_model_from_config,
    register_model,
)

# baseline 模型：导入模块即完成注册。
from csifeedback.models.clnet import CLNet, create_clnet
from csifeedback.models.csinet import CsiNet, create_csinet
from csifeedback.models.stnet import STNet, create_stnet


__all__ = [
    "CSIAutoencoder",
    "CLNet",
    "STNet",
    "CsiNet",
    "create_clnet",
    "create_stnet",
    "create_csinet",
    "register_model",
    "get_model",
    "get_model_from_config",
    "available_models",
    "_MODEL_FACTORIES",
]
