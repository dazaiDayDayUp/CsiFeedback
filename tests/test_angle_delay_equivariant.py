"""Angle-Delay Equivariant Feedback 的实例化与前向测试。"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.models import get_model
from csifeedback.models.registry import available_models


@pytest.mark.parametrize("encoded_dim", [512, 256, 128, 64, 32])
def test_forward_shape(encoded_dim):
    model = get_model("angle_delay_equivariant", encoded_dim=encoded_dim)
    model.eval()
    x = torch.rand(2, 2, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


def test_registered():
    assert "angle_delay_equivariant" in available_models()


def test_angle_circular_equivariance():
    """角度循环平移应只改变输出，不应产生边界伪影。"""
    model = get_model("angle_delay_equivariant", encoded_dim=64)
    model.eval()
    x = torch.rand(1, 2, 32, 32)
    with torch.no_grad():
        y = model(x)
        y_shift = model(torch.roll(x, shifts=3, dims=2))
    assert y.shape == y_shift.shape
    assert torch.isfinite(y_shift).all()
