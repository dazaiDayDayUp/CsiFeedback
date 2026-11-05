"""ADEF 专用训练器：在标准 MSE 上加梯度裁剪，抑制注意力不稳定。"""

import torch

from csifeedback.trainers.default import DefaultTrainer


__all__ = ["AngleDelayEquivariantTrainer"]


class AngleDelayEquivariantTrainer(DefaultTrainer):
    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        self.grad_clip = float(config.model.extra.get("grad_clip", 1.0))

    def _train_step(self, x):
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, x)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()
