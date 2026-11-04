"""Path-Geometry Feedback 专用训练器。

训练分两阶段：
1. 路径预训练：只用“仅多径合成”的输出重建，让热图检测器和路径分支先学会
   specular 几何；残差分支不参与损失。
2. 联合微调：最终重建 + 路径辅助重建 + 漫散射残差。

两个阶段都使用热图/增益图的物理辅助损失：
- 热图：目标为输入角-时延幅度 top-K 峰的高斯热图；
- 增益图：在目标峰位置回归复增益。
"""

import torch
import torch.nn.functional as F

from csifeedback.trainers.default import DefaultTrainer


__all__ = ["PathGeometryFeedbackTrainer"]


class PathGeometryFeedbackTrainer(DefaultTrainer):
    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        extra = config.model.extra
        self.path_loss_weight = float(extra.get("path_loss_weight", 0.2))
        self.path_pretrain_epochs = int(extra.get("path_pretrain_epochs", 150))
        self.path_pretrain_loss_weight = float(
            extra.get("path_pretrain_loss_weight", 1.0)
        )
        self.heatmap_loss_weight = float(
            extra.get("heatmap_loss_weight", 0.1)
        )
        self.gain_loss_weight = float(extra.get("gain_loss_weight", 0.1))
        self.heatmap_sigma = float(extra.get("heatmap_sigma", 0.7))
        self.heatmap_pos_weight = float(
            extra.get("heatmap_pos_weight", 10.0)
        )
        self.code_loss_weight = float(extra.get("code_loss_weight", 0.1))

    def _target_maps(self, x: torch.Tensor):
        n = x.shape[0]
        csi = (x[:, 0] - 0.5) + 1j * (x[:, 1] - 0.5)
        magnitude = csi.abs().flatten(1)
        _, index = torch.topk(magnitude, self.model.num_paths, dim=1)
        angle0 = (index // 32).float()
        delay0 = (index % 32).float()

        aa = torch.arange(32, device=x.device, dtype=torch.float32).view(
            1, 1, 32, 1
        )
        dd = torch.arange(32, device=x.device, dtype=torch.float32).view(
            1, 1, 1, 32
        )
        dist2 = (aa - angle0.view(n, -1, 1, 1)) ** 2 + (
            dd - delay0.view(n, -1, 1, 1)
        ) ** 2
        heatmap = torch.exp(
            -dist2 / (2.0 * self.heatmap_sigma ** 2)
        ).max(dim=1).values.unsqueeze(1)

        target_real = csi.real.flatten(1).gather(1, index)
        target_imag = csi.imag.flatten(1).gather(1, index)
        return heatmap, index, target_real, target_imag

    def _aux_losses(self, x, heatmap, gain_map):
        target_heatmap, index, target_real, target_imag = self._target_maps(x)
        heat_loss = F.binary_cross_entropy_with_logits(
            heatmap,
            target_heatmap,
            pos_weight=torch.tensor(
                [self.heatmap_pos_weight], device=x.device
            ),
        )
        gain_pred = gain_map.flatten(2).gather(
            2, index.unsqueeze(1).expand(-1, 2, -1)
        )
        gain_loss = F.mse_loss(
            gain_pred[:, 0], target_real
        ) + F.mse_loss(gain_pred[:, 1], target_imag)
        return heat_loss, gain_loss

    def _greedy_match(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """为每个目标峰匹配最近的预测路径，返回匹配后的预测。"""
        n, k, _ = pred.shape
        pred_angle = pred[..., 0] * 32.0
        pred_delay = pred[..., 1] * self.model.decoder.delay_fft_size
        target_angle = target[..., 0] * 32.0
        target_delay = target[..., 1] * self.model.decoder.delay_fft_size
        cost = (pred_angle.unsqueeze(2) - target_angle.unsqueeze(1)).pow(2)
        cost = cost + (
            pred_delay.unsqueeze(2) - target_delay.unsqueeze(1)
        ).pow(2)
        cost = cost + (
            pred[..., 2:].unsqueeze(2) - target[..., 2:].unsqueeze(1)
        ).pow(2).sum(dim=-1)

        available = torch.ones(n, k, dtype=torch.bool, device=pred.device)
        row = torch.arange(n, device=pred.device)
        matched = []
        for j in range(k):
            current = cost[:, :, j].masked_fill(~available, 1e9)
            choice = current.argmin(dim=1)
            matched.append(choice)
            available[row, choice] = False
        choice = torch.stack(matched, dim=1)
        return pred.gather(1, choice.unsqueeze(-1).expand(-1, -1, 4))

    def _code_alignment_loss(self, x, code):
        num_paths = self.model.num_paths
        pred = code[:, : num_paths * 4].view(-1, num_paths, 4)
        csi = (x[:, 0] - 0.5) + 1j * (x[:, 1] - 0.5)
        _, index = torch.topk(
            csi.abs().flatten(1), num_paths, dim=1
        )
        target = torch.stack(
            (
                (index // 32).float() / 32.0,
                (index % 32).float()
                / self.model.decoder.delay_fft_size,
                csi.real.flatten(1).gather(1, index),
                csi.imag.flatten(1).gather(1, index),
            ),
            dim=-1,
        )
        matched = self._greedy_match(pred, target)
        return (
            F.mse_loss(matched[..., 0], target[..., 0])
            + F.mse_loss(
                matched[..., 1] * 32.0, target[..., 1] * 32.0
            )
            + F.mse_loss(matched[..., 2], target[..., 2])
            + F.mse_loss(matched[..., 3], target[..., 3])
        )

    def _train_step(self, x):
        self.optimizer.zero_grad()
        code, heatmap, gain_map = self.model.encoder(
            x, return_maps=True
        )
        out, path_out = self.model.decoder(code)

        if self.epoch < self.path_pretrain_epochs:
            loss = self.path_pretrain_loss_weight * self.criterion(
                path_out, x
            )
        else:
            loss = self.criterion(out, x) + self.path_loss_weight * self.criterion(
                path_out, x
            )

        heat_loss, gain_loss = self._aux_losses(x, heatmap, gain_map)
        code_loss = self._code_alignment_loss(x, code)
        loss = (
            loss
            + self.heatmap_loss_weight * heat_loss
            + self.gain_loss_weight * gain_loss
            + self.code_loss_weight * code_loss
        )
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()
