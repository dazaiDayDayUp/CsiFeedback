"""STNet 训练器：分别为 encoder/decoder 使用独立的 Adam 优化器。"""

import os
from typing import Any, Dict, Optional

import torch
from torch.optim import Adam

from csifeedback.models.base import CSIAutoencoder
from csifeedback.trainers.base import BaseTrainer
from csifeedback.utils.checkpoint import load_checkpoint
from csifeedback.utils.config import ExperimentConfig
from csifeedback.utils.scheduler import FakeLR


__all__ = ["STNetTrainer"]


class STNetTrainer(BaseTrainer):
    """STNet 训练器。

    为 encoder 和 decoder 分别使用 Adam 优化器，二者 betas 均为 ``(0.5, 0.999)``。
    检查点以分开的 encoder/decoder state dict 保存，以保持与原始格式兼容。
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: CSIAutoencoder,
        device: torch.device,
    ):
        if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
            raise ValueError("STNetTrainer 要求模型具有 'encoder' 和 'decoder' 属性")

        self.opt_enc = Adam(
            model.encoder.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
        self.opt_dec = Adam(
            model.decoder.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )

        # 用一个占位优化器满足基类 API。
        self._dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scheduler = FakeLR(self._dummy_optimizer)

        super().__init__(config, model, device, self._dummy_optimizer, scheduler)

    def _get_current_lr(self) -> float:
        """返回 encoder 优化器的当前学习率（enc/dec 使用相同 lr）。"""
        return self.opt_enc.param_groups[0]["lr"]

    def _train_step(self, x: torch.Tensor) -> float:
        self.opt_enc.zero_grad()
        self.opt_dec.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, x)
        loss.backward()
        self.opt_enc.step()
        self.opt_dec.step()
        return loss.item()

    def _get_optimizer_state_dict(self) -> Dict[str, Any]:
        return {
            "opt_enc": self.opt_enc.state_dict(),
            "opt_dec": self.opt_dec.state_dict(),
        }

    def _get_model_state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.model.get_encoder_state(),
            "decoder": self.model.get_decoder_state(),
        }

    def _maybe_resume(self) -> None:
        resume_path = self.training_config.resume
        if resume_path is None:
            return
        ckpt = load_checkpoint(resume_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt.model_state_dict)
        opt_state = ckpt.optimizer_state_dict
        if isinstance(opt_state, dict) and "opt_enc" in opt_state:
            self.opt_enc.load_state_dict(opt_state["opt_enc"])
            self.opt_dec.load_state_dict(opt_state["opt_dec"])
        self.epoch = ckpt.epoch
        self.best_rho = ckpt.best_rho if ckpt.best_rho is not None else float("-inf")
        self.best_nmse = ckpt.best_nmse if ckpt.best_nmse is not None else float("inf")
        self.logger.info("STNet 从检查点 %s 恢复，当前 epoch %d", resume_path, self.epoch)

    def _save_checkpoint(self, filename: str) -> None:
        # STNet 原始实现分别保存 encoder/decoder 检查点。
        path = os.path.join(self.exp_dir, "checkpoints", filename)
        from csifeedback.utils.checkpoint import save_checkpoint
        save_checkpoint(
            path=path,
            epoch=self.epoch + 1,
            model_name=self.config.model.name,
            config=self.config,
            model_state_dict=self._get_model_state_dict(),
            optimizer_state_dict=self._get_optimizer_state_dict(),
            scheduler_state_dict=None,
            best_rho=self.best_rho,
            best_nmse=self.best_nmse,
        )
