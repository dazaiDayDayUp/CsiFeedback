"""Path-Geometry Feedback 专用训练器。

三项损失：
1. 最终重建 MSE；
2. 仅多径合成的辅助重建 MSE，保证路径头有梯度；
3. 角-时延主峰对齐损失：把路径 token 拉向输入的 top-K 幅度峰，
   解决随机初始化时连续频率梯度过小、路径分支停在零附近的平台。
峰对齐损失在前若干 epoch 后退火到 0，之后由重建损失细化连续参数。
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
        self.peak_loss_weight = float(extra.get("peak_loss_weight", 0.005))
        self.peak_loss_warmup = int(extra.get("peak_loss_warmup", 150))
        self.peak_loss_decay = int(extra.get("peak_loss_decay", 250))

    def _peak_weight(self) -> float:
        if self.peak_loss_weight <= 0:
            return 0.0
        epoch = self.epoch
        end = self.peak_loss_warmup + self.peak_loss_decay
        if epoch < self.peak_loss_warmup:
            return self.peak_loss_weight
        if epoch < end:
            return self.peak_loss_weight * (end - epoch) / max(1, self.peak_loss_decay)
        return 0.0

    def _peak_alignment_loss(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        num_paths = self.model.num_paths
        params = code[:, : num_paths * 4].view(-1, num_paths, 4)
        angle = params[..., 0]
        delay = params[..., 1]
        gain = params[..., 2:4]

        # 按增益幅度排序，与按幅度排序的 top-K 峰逐条对应。
        order = torch.argsort(gain.norm(dim=-1), dim=1, descending=True)
        angle = angle.gather(1, order)
        delay = delay.gather(1, order)
        gain = gain.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))

        csi = (x[:, 0] - 0.5) + 1j * (x[:, 1] - 0.5)
        magnitude = csi.abs().flatten(1)
        _, index = torch.topk(magnitude, num_paths, dim=1)

        target_angle = (index // 32).float() / 32.0
        target_delay = (index % 32).float() / self.model.decoder.delay_fft_size
        target_real = csi.real.flatten(1).gather(1, index)
        target_imag = csi.imag.flatten(1).gather(1, index)

        # delay 用 tap 尺度（0..32）比较，避免归一化后信号过小。
        return (
            F.mse_loss(angle, target_angle)
            + F.mse_loss(
                delay * 32.0, target_delay * 32.0
            )
            + F.mse_loss(gain[..., 0], target_real)
            + F.mse_loss(gain[..., 1], target_imag)
        )

    def _train_step(self, x):
        self.optimizer.zero_grad()
        code = self.model.encoder(x)
        out, path_out = self.model.decoder(code)
        loss = self.criterion(out, x) + self.path_loss_weight * self.criterion(
            path_out, x
        )
        peak_weight = self._peak_weight()
        if peak_weight > 0:
            loss = loss + peak_weight * self._peak_alignment_loss(x, code)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()
