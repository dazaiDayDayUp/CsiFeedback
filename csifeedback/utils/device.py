"""设备初始化与可复现性辅助函数。"""

import os
import random
from typing import Optional

import numpy as np
import torch


__all__ = ["setup_device", "set_seed"]


def setup_device(device_name: Optional[str] = None) -> torch.device:
    """返回 torch.device 并配置后端默认行为。

    参数：
        device_name: ``"auto"``、``"cpu"``、``"cuda"`` 或 ``"cuda:N"``。
            ``None`` 与 ``"auto"`` 都会在 CUDA 可用时选择 CUDA。
    """
    if device_name is None or device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def set_seed(seed: Optional[int]) -> None:
    """设置随机种子以保证可复现性。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性算法可能降低性能；为速度保持 benchmark。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
