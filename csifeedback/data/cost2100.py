"""CLNet、STNet、CsiNet 统一的 COST2100 数据集加载器。"""

import os
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset


__all__ = ["Cost2100Dataset", "get_cost2100_loaders"]


def _normalize_scenario(scenario: str) -> str:
    """同时兼容 CLNet 风格（'in'/'out'）和 STNet/CsiNet 风格（'indoor'/'outdoor'）。"""
    mapping = {
        "in": "in",
        "out": "out",
        "indoor": "in",
        "outdoor": "out",
    }
    key = scenario.lower().strip()
    if key not in mapping:
        raise ValueError(f"scenario 必须是 {set(mapping.keys())} 之一，得到 {scenario}")
    return mapping[key]


class Cost2100Dataset(TensorDataset):
    """COST2100 稀疏 CSI 数据集。

    从 ``DATA_H{split}{scenario}.mat`` 加载键 ``HT``，形状为 ``(N, 2, 32, 32)``。
    对于测试集，可选择同时加载频域原始 CSI 键 ``HF_all``，形状为 ``(N, 32, 125, 2)``。
    """

    channel: int = 2
    nt: int = 32
    nc: int = 32
    nc_expand: int = 125

    def __init__(
        self,
        data_dir: str,
        scenario: str,
        split: str,
        load_raw: bool = False,
    ) -> None:
        data_dir = os.path.abspath(data_dir)
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"data_dir 不存在：{data_dir}")

        scenario = _normalize_scenario(scenario)
        split = split.lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split 必须是 train/val/test 之一，得到 {split}")

        mat_path = os.path.join(data_dir, f"DATA_H{split}{scenario}.mat")
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"数据集文件不存在：{mat_path}")

        sparse = sio.loadmat(mat_path)["HT"]
        sparse = torch.tensor(sparse, dtype=torch.float32).view(
            sparse.shape[0], self.channel, self.nt, self.nc
        )

        if split == "test" and load_raw:
            raw_path = os.path.join(data_dir, f"DATA_HtestF{scenario}_all.mat")
            if not os.path.isfile(raw_path):
                raise FileNotFoundError(f"原始 CSI 文件不存在：{raw_path}")
            raw_test = sio.loadmat(raw_path)["HF_all"]
            real = torch.tensor(np.real(raw_test), dtype=torch.float32)
            imag = torch.tensor(np.imag(raw_test), dtype=torch.float32)
            raw = torch.cat(
                (
                    real.view(raw_test.shape[0], self.nt, self.nc_expand, 1),
                    imag.view(raw_test.shape[0], self.nt, self.nc_expand, 1),
                ),
                dim=3,
            )
            super().__init__(sparse, raw)
        else:
            super().__init__(sparse)


def get_cost2100_loaders(
    data_dir: str,
    scenario: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    load_raw: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """返回 COST2100 的 (训练加载器, 验证加载器, 测试加载器)。"""
    train_ds = Cost2100Dataset(data_dir, scenario, "train")
    val_ds = Cost2100Dataset(data_dir, scenario, "val")
    test_ds = Cost2100Dataset(data_dir, scenario, "test", load_raw=load_raw)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
