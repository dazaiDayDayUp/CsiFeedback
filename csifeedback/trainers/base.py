"""所有 CSI 反馈模型共享的训练器基类。"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import tqdm
from colorama import Fore, Style, init
from torch.optim import Optimizer
from torch.utils.data import DataLoader

init(autoreset=True)

from csifeedback.metrics.csi_metrics import compute_nmse, compute_rho
from csifeedback.models.base import CSIAutoencoder
from csifeedback.utils.checkpoint import load_checkpoint, save_checkpoint
from csifeedback.utils.config import ExperimentConfig, TrainingConfig
from csifeedback.utils.logging import get_logger


__all__ = ["BaseTrainer"]


class BaseTrainer:
    """通用训练循环，包含验证、测试和检查点保存。"""

    def __init__(
        self,
        config: ExperimentConfig,
        model: CSIAutoencoder,
        device: torch.device,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.training_config: TrainingConfig = config.training
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.MSELoss()
        self.logger = get_logger(self.__class__.__name__)

        self.epoch = 0
        self.global_step = 0
        self.best_rho = float("-inf")
        self.best_nmse = float("inf")
        self.best_val_loss = float("inf")

        self.exp_dir = self._make_exp_dir()
        self._setup_logging()

    def _make_exp_dir(self) -> str:
        """创建带场景、压缩比和时间戳的实验目录。"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model.name
        scenario = self.config.data.scenario
        cr_label = self._cr_label()

        exp_dir = os.path.join(
            self.training_config.checkpoint_dir,
            model_name,
            f"{scenario}_{cr_label}",
            timestamp,
        )
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        return exp_dir

    def _cr_label(self) -> str:
        """根据模型配置返回压缩比标签，例如 cr4、cr8。"""
        model_name = self.config.model.name
        if model_name == "clnet":
            reduction = self.config.model.reduction
            if reduction is None:
                raise ValueError("CLNet 配置缺少 model.reduction")
            return f"cr{reduction}"
        elif model_name in {"stnet", "csinet"}:
            encoded_dim = self.config.model.encoded_dim
            if encoded_dim is None:
                raise ValueError(f"{model_name} 配置缺少 model.encoded_dim")
            reduction = 2048 // encoded_dim
            return f"cr{reduction}"
        else:
            return "unknown_cr"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """将秒数格式化为 HH:MM:SS。"""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _setup_logging(self) -> None:
        from csifeedback.utils.logging import setup_logging
        log_file = os.path.join(self.exp_dir, "logs", "train.log")
        setup_logging(log_file=log_file)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """运行完整训练循环。"""
        self._maybe_resume()
        start_epoch = self.epoch
        epochs = self.training_config.epochs

        self.logger.info(
            "开始训练 %s，从 epoch %d 到 %d，设备 %s",
            self.config.model.name,
            start_epoch + 1,
            epochs,
            self.device,
        )

        train_start_time = time.time()

        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            train_loss = self._train_epoch(train_loader)
            epoch_duration = time.time() - epoch_start_time
            total_elapsed = time.time() - train_start_time
            completed_epochs = epoch - start_epoch + 1
            remaining_epochs = epochs - epoch - 1
            avg_epoch_time = total_elapsed / max(1, completed_epochs)
            eta = avg_epoch_time * remaining_epochs

            self.logger.info(
                "Epoch [%d/%d] train_loss: %.4e  本轮 %s  累计 %s  预计剩余 %s",
                epoch + 1,
                epochs,
                train_loss,
                self._format_time(epoch_duration),
                self._format_time(total_elapsed),
                self._format_time(eta),
            )

            if val_loader is not None and self._is_val_epoch(epoch):
                val_loss, val_nmse = self._validate(val_loader)
                self.logger.info(
                    "Epoch [%d/%d] val_loss: %.4e  val_NMSE: %.4f dB",
                    epoch + 1,
                    epochs,
                    val_loss,
                    val_nmse,
                )
                self._maybe_save_best(val_loss)

            if test_loader is not None and self._is_test_epoch(epoch):
                test_loss, rho, nmse = self._test(test_loader)
                self.logger.info(
                    "Epoch [%d/%d] test_loss: %.4e  rho: %.4f  NMSE: %.4f dB",
                    epoch + 1,
                    epochs,
                    test_loss,
                    rho,
                    nmse,
                )
                self._maybe_update_best_metrics(rho, nmse)

            self._save_periodic_checkpoint(epoch)

        self._save_last_checkpoint()
        self.logger.info("训练结束。实验目录：%s", self.exp_dir)

    def _get_current_lr(self) -> float:
        """返回当前学习率。子类（如 STNet）可针对多个优化器重写。"""
        return self.optimizer.param_groups[0]["lr"]

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        desc = f"{Fore.CYAN}Epoch {self.epoch + 1}/{self.training_config.epochs}{Style.RESET_ALL}"
        pbar = tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=desc,
            leave=True,
            unit="batch",
            colour="green",
        )

        for batch_idx, batch in pbar:
            x = batch[0].to(self.device, non_blocking=True)
            loss = self._train_step(x)
            total_loss += loss
            n_batches += 1
            self.global_step += 1

            lr = self._get_current_lr()
            pbar.set_postfix({
                f"{Fore.YELLOW}loss{Style.RESET_ALL}": f"{loss:.4e}",
                f"{Fore.MAGENTA}lr{Style.RESET_ALL}": f"{lr:.6f}",
            })

            if self.training_config.print_freq > 0 and (
                batch_idx % self.training_config.print_freq == 0
            ):
                self.logger.debug(
                    "Epoch [%d/%d] Batch [%d/%d] loss: %.4e lr: %.6f",
                    self.epoch + 1,
                    self.training_config.epochs,
                    batch_idx,
                    len(train_loader),
                    loss,
                    lr,
                )

        return total_loss / max(1, n_batches)

    def _train_step(self, x: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, x)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_pred = []
        all_gt = []

        for batch in val_loader:
            x = batch[0].to(self.device, non_blocking=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            total_loss += loss.item()
            n_batches += 1
            all_pred.append(out.cpu())
            all_gt.append(x.cpu())

        preds = torch.cat(all_pred, dim=0)
        gts = torch.cat(all_gt, dim=0)
        nmse = compute_nmse(preds, gts).item()
        return total_loss / max(1, n_batches), nmse

    @torch.no_grad()
    def _test(self, test_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_pred = []
        all_gt = []
        all_raw = []

        for batch in test_loader:
            if len(batch) == 2:
                x, raw = batch
                all_raw.append(raw)
            else:
                x = batch[0]
            x = x.to(self.device, non_blocking=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            total_loss += loss.item()
            n_batches += 1
            all_pred.append(out.cpu())
            all_gt.append(x.cpu())

        preds = torch.cat(all_pred, dim=0)
        gts = torch.cat(all_gt, dim=0)
        nmse = compute_nmse(preds, gts).item()

        if all_raw:
            raw = torch.cat(all_raw, dim=0)
            rho = compute_rho(preds, raw).item()
        else:
            rho = float("nan")

        return total_loss / max(1, n_batches), rho, nmse

    def _is_val_epoch(self, epoch: int) -> bool:
        freq = self.training_config.val_freq
        return freq > 0 and (epoch + 1) % freq == 0

    def _is_test_epoch(self, epoch: int) -> bool:
        freq = self.training_config.test_freq
        return freq > 0 and (epoch + 1) % freq == 0

    def _maybe_update_best_metrics(self, rho: float, nmse: float) -> None:
        if rho > self.best_rho:
            self.best_rho = rho
            self._save_checkpoint("best_rho.pth")
        if nmse < self.best_nmse:
            self.best_nmse = nmse
            self._save_checkpoint("best_nmse.pth")

    def _maybe_save_best(self, val_loss: float) -> None:
        """钩子，供模型特定的最优保存逻辑使用（例如 CsiNet）。"""
        pass

    def _save_periodic_checkpoint(self, epoch: int) -> None:
        freq = self.training_config.checkpoint_freq
        if freq > 0 and (epoch + 1) % freq == 0:
            self._save_checkpoint(f"epoch{epoch + 1}.pth")

    def _save_last_checkpoint(self) -> None:
        self._save_checkpoint("last.pth")

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.exp_dir, "checkpoints", filename)
        save_checkpoint(
            path=path,
            epoch=self.epoch + 1,
            model_name=self.config.model.name,
            config=self.config,
            model_state_dict=self._get_model_state_dict(),
            optimizer_state_dict=self._get_optimizer_state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
            best_rho=self.best_rho,
            best_nmse=self.best_nmse,
        )

    def _get_model_state_dict(self) -> Dict[str, Any]:
        """返回模型状态。子类可针对分开的 encoder/decoder 重写。"""
        return self.model.state_dict()

    def _get_optimizer_state_dict(self) -> Any:
        """返回优化器状态。子类可针对多个优化器重写。"""
        return self.optimizer.state_dict()

    def _maybe_resume(self) -> None:
        resume_path = self.training_config.resume
        if resume_path is None:
            return
        ckpt = load_checkpoint(resume_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt.model_state_dict)
        self.optimizer.load_state_dict(ckpt.optimizer_state_dict)
        if self.scheduler is not None and ckpt.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(ckpt.scheduler_state_dict)
        self.epoch = ckpt.epoch
        self.best_rho = ckpt.best_rho if ckpt.best_rho is not None else float("-inf")
        self.best_nmse = ckpt.best_nmse if ckpt.best_nmse is not None else float("inf")
        self.logger.info("从检查点 %s 恢复，当前 epoch %d", resume_path, self.epoch)
