"""NMSE 与 rho 指标正确性的测试。"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.metrics.csi_metrics import compute_nmse, compute_rho


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


def test_nmse_matches_legacy_clnet():
    """在随机稀疏张量上与原始 CLNet evaluator 进行比较。"""
    import importlib.util

    legacy_path = repo_root / "archive/CLNet/utils/statics.py"
    if not legacy_path.exists():
        pytest.skip("未找到原始 statics.py")

    spec = importlib.util.spec_from_file_location("legacy_statics", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    torch.manual_seed(0)
    pred = torch.rand(8, 2, 32, 32)
    gt = torch.rand(8, 2, 32, 32)
    raw = torch.rand(8, 32, 125, 2)

    rho_new, nmse_new = legacy.evaluator(pred, gt, raw)
    nmse_ours = compute_nmse(pred, gt)
    rho_ours = compute_rho(pred, raw)

    assert torch.isclose(nmse_ours, nmse_new, atol=1e-6)
    assert torch.isclose(rho_ours, rho_new, atol=1e-6)
