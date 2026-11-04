"""Path-Geometry Feedback 的实例化、前向与配置测试。"""

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
    model = get_model("path_geometry_feedback", encoded_dim=encoded_dim)
    model.eval()
    x = torch.rand(2, 2, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.parametrize("encoded_dim", [512, 64])
def test_return_path(encoded_dim):
    model = get_model("path_geometry_feedback", encoded_dim=encoded_dim)
    model.eval()
    x = torch.rand(3, 2, 32, 32)
    with torch.no_grad():
        out, path_out = model(x, return_path=True)
    assert out.shape == x.shape
    assert path_out.shape == x.shape


def test_registered():
    assert "path_geometry_feedback" in available_models()


def test_encoded_dim_property():
    model = get_model("path_geometry_feedback", encoded_dim=256)
    assert model.encoded_dim == 256


def test_path_gradient_flows():
    model = get_model("path_geometry_feedback", encoded_dim=512)
    model.train()
    x = torch.rand(2, 2, 32, 32)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, x)
    loss.backward()
    grads = [
        p.grad.abs().sum().item()
        for p in model.encoder.path_head.parameters()
        if p.grad is not None
    ]
    assert sum(grads) > 0
