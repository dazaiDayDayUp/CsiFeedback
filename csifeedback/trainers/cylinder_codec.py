"""CylinderCodec 训练器：逐样本归一化 NMSE 目标 + 梯度裁剪。

baseline 用逐元素 MSE 训练，但报告指标是

    NMSE = 10·log10( mean_over_samples( ||pred-gt||² / ||gt-0.5||² ) )

两者对样本的加权不同：MSE 按功率加权（强样本主导），NMSE 按 1/功率加权
（弱样本与强样本等权，且动态聚焦当前误差大的样本）。这里直接用与报告指标
完全一致的逐样本归一化目标训练，使优化目标与部署指标对齐。
"""

import torch
import torch.nn as nn

from csifeedback.trainers.default import DefaultTrainer
from csifeedback.trainers.stnet import STNetTrainer


__all__ = ["MetricMSELoss", "CylinderCodecTrainer", "STNetMetricTrainer"]


class MetricMSELoss(nn.Module):
    """逐样本归一化均方误差（与 ``compute_nmse`` 同一约定，不含 log）。"""

    def __init__(self, zero_point: float = 0.5, min_power: float = 1e-3):
        super().__init__()
        self.zero_point = zero_point
        self.min_power = min_power

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target - self.zero_point
        p = pred - self.zero_point
        power = (t[:, 0] ** 2 + t[:, 1] ** 2).sum(dim=(1, 2))
        diff = p - t
        mse = (diff[:, 0] ** 2 + diff[:, 1] ** 2).sum(dim=(1, 2))
        return (mse / power.clamp_min(self.min_power)).mean()


class CylinderCodecTrainer(DefaultTrainer):
    """使用逐样本归一化目标的配置驱动训练器。

    ``model.extra.loss`` 可选 ``metric``（默认）或 ``mse``，用于目标函数消融；
    ``model.extra.grad_clip`` 控制梯度裁剪范数（0 表示关闭）。
    """

    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        loss_name = str(config.model.extra.get("loss", "metric")).lower()
        if loss_name == "metric":
            self.criterion = MetricMSELoss()
        elif loss_name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"model.extra.loss 必须是 metric/mse，得到 {loss_name!r}")
        self.grad_clip = float(config.model.extra.get("grad_clip", 1.0))

    def _train_step(self, x: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, x)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()


class STNetMetricTrainer(STNetTrainer):
    """STNet 的双优化器训练循环 + 逐样本归一化目标（公平性对照）。"""

    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        self.criterion = MetricMSELoss()
