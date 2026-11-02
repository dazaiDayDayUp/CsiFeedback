"""NMSE 与 rho 指标正确性的测试。"""

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.metrics.csi_metrics import compute_nmse


def test_nmse_perfect_reconstruction():
    x = torch.rand(4, 2, 32, 32)
    nmse = compute_nmse(x, x)
    # 完美重建时 MSE 为 0，NMSE 应为 -inf dB。
    assert torch.isinf(nmse) and nmse < 0


def test_nmse_known_value():
    # pred 全为 0.5，去中心化后为零
    # gt   全为 0.6，去中心化后全为 0.1
    # 功率与误差幅度相同，NMSE = 0 dB
    pred = torch.full((1, 2, 32, 32), 0.5)
    gt = torch.full((1, 2, 32, 32), 0.6)
    nmse = compute_nmse(pred, gt)
    assert torch.isclose(nmse, torch.tensor(0.0), atol=1e-5)
