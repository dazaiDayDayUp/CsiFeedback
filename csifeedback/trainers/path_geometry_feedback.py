"""Path-Geometry Feedback 专用训练器。

在标准 MSE 之外，对“仅多径合成”的输出加一个辅助重建损失，避免残差分支
独自完成重建、路径头学不到可解释的几何。
"""

from csifeedback.trainers.default import DefaultTrainer


__all__ = ["PathGeometryFeedbackTrainer"]


class PathGeometryFeedbackTrainer(DefaultTrainer):
    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        self.path_loss_weight = float(
            config.model.extra.get("path_loss_weight", 0.2)
        )

    def _train_step(self, x):
        self.optimizer.zero_grad()
        out, path_out = self.model(x, return_path=True)
        loss = self.criterion(out, x) + self.path_loss_weight * self.criterion(
            path_out, x
        )
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()
