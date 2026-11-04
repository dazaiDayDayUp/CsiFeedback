"""Angular-Spectral Feedback 的实例化与前向测试。"""

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
    model = get_model("angular_spectral_feedback", encoded_dim=encoded_dim)
    model.eval()
    x = torch.rand(2, 2, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


def test_registered():
    assert "angular_spectral_feedback" in available_models()


def test_spectral_block_shape():
    from csifeedback.models.angular_spectral_feedback import SpectralBlock

    block = SpectralBlock(64, angle_size=8, modes=5)
    x = torch.rand(2, 64, 8, 8)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
