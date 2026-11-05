"""CylinderCodec 训练器：逐样本归一化 NMSE 目标 + 梯度裁剪。

baseline 用逐元素 MSE 训练，但报告指标是

    NMSE = 10·log10( mean_over_samples( ||pred-gt||² / ||gt-0.5||² ) )

两者对样本的加权不同：MSE 按功率加权（强样本主导），NMSE 按 1/功率加权
（弱样本与强样本等权，且动态聚焦当前误差大的样本）。这里直接用与报告指标
完全一致的逐样本归一化目标训练，使优化目标与部署指标对齐。
"""

import torch
import torch.nn as nn

from csifeedback.metrics.csi_metrics import compute_nmse
from csifeedback.trainers.default import DefaultTrainer
from csifeedback.trainers.stnet import STNetTrainer


__all__ = ["MetricMSELoss", "CylinderCodecTrainer", "STNetMetricTrainer"]


class MetricMSELoss(nn.Module):
    """逐样本归一化均方误差（与 ``compute_nmse`` 同一约定，不含 log）。

    ``ratio_clamp``：单样本 ratio 的上限。ratio > 1 意味着重建误差已大于信号本身
    （训练早期的灾难样本），若不设上限，单个样本会主导整个 batch 的梯度并造成
    训练振荡。默认 10.0，仅在病态样本上生效，不改变正常区间的目标。
    """

    def __init__(self, zero_point: float = 0.5, min_power: float = 1e-3,
                 ratio_clamp=None):
        super().__init__()
        self.zero_point = zero_point
        self.min_power = min_power
        self.ratio_clamp = ratio_clamp

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target - self.zero_point
        p = pred - self.zero_point
        power = (t[:, 0] ** 2 + t[:, 1] ** 2).sum(dim=(1, 2))
        diff = p - t
        mse = (diff[:, 0] ** 2 + diff[:, 1] ** 2).sum(dim=(1, 2))
        ratio = mse / power.clamp_min(self.min_power)
        if self.ratio_clamp:
            ratio = ratio.clamp_max(self.ratio_clamp)
        return ratio.mean()


class CylinderCodecTrainer(DefaultTrainer):
    """使用逐样本归一化目标的配置驱动训练器。

    ``model.extra.loss`` 可选 ``metric``（默认）或 ``mse``，用于目标函数消融；
    ``model.extra.grad_clip`` 控制梯度裁剪范数（0 表示关闭）。
    """

    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        clamp = config.model.extra.get("ratio_clamp", None)
        if clamp is not None:
            clamp = float(clamp)
        loss_name = str(config.model.extra.get("loss", "metric")).lower()
        if loss_name == "metric":
            self.criterion = MetricMSELoss(ratio_clamp=clamp)
        elif loss_name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"model.extra.loss 必须是 metric/mse，得到 {loss_name!r}")
        self.grad_clip = float(config.model.extra.get("grad_clip", 1.0))
        # 选模型始终用逐样本归一化指标，即使训练目标是 MSE —— 保证目标函数消融只
        # 改变训练信号，不改变 checkpoint 选择准则。
        self.val_criterion = MetricMSELoss(ratio_clamp=clamp)

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        total = 0.0
        n_batches = 0
        all_pred = []
        all_gt = []
        for batch in val_loader:
            x = batch[0].to(self.device, non_blocking=True)
            out = self.model(x)
            total += self.val_criterion(out, x).item()
            n_batches += 1
            all_pred.append(out.cpu())
            all_gt.append(x.cpu())
        preds = torch.cat(all_pred, dim=0)
        gts = torch.cat(all_gt, dim=0)
        return total / max(1, n_batches), compute_nmse(preds, gts).item()

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
