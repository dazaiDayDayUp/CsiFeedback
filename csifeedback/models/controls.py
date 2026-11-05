"""公平性对照模型。

只用于消融/公平性论证：把 baseline 网络包一层，使其可以用与
``CylinderCodec`` 完全相同的训练目标、调度与 checkpoint 选择策略训练，
从而把「新目标带来的增益」与「新结构带来的增益」分开。
"""

from csifeedback.models.registry import register_model
from csifeedback.models.stnet import STNet


__all__ = ["create_stnet_metric"]


@register_model("stnet_metric")
def create_stnet_metric(encoded_dim: int = 512, **kwargs) -> STNet:
    """STNet 网络 + 逐样本归一化目标（训练器见 ``STNetMetricTrainer``）。"""
    return STNet(encoded_dim=encoded_dim)
