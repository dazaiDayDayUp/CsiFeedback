"""统一的检查点保存与加载工具。"""

import os
import time
from typing import Any, Dict, Optional

import torch

from csifeedback.utils.config import ExperimentConfig, config_to_dict


__all__ = ["save_checkpoint", "load_checkpoint", "Checkpoint"]


class Checkpoint(Dict[str, Any]):
    """检查点字典的便捷包装类。"""

    @property
    def epoch(self) -> int:
        return self.get("epoch", 0)

    @property
    def model_name(self) -> str:
        return self.get("model_name", "")

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        return self.get("config")

    @property
    def model_state_dict(self) -> Dict[str, Any]:
        return self.get("model_state_dict", {})

    @property
    def optimizer_state_dict(self) -> Any:
        return self.get("optimizer_state_dict")

    @property
    def scheduler_state_dict(self) -> Any:
        return self.get("scheduler_state_dict")

    @property
    def best_rho(self) -> Optional[float]:
        return self.get("best_rho")

    @property
    def best_nmse(self) -> Optional[float]:
        return self.get("best_nmse")


def save_checkpoint(
    path: str,
    epoch: int,
    model_name: str,
    config: ExperimentConfig,
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Any,
    scheduler_state_dict: Optional[Any] = None,
    best_rho: Optional[float] = None,
    best_nmse: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存包含完整元信息的统一格式检查点。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_name": model_name,
        "config": config_to_dict(config),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "best_rho": best_rho,
        "best_nmse": best_nmse,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Checkpoint:
    """加载统一格式检查点。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点不存在：{path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return Checkpoint(ckpt)
