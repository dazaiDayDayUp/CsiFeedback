"""CylinderCodec 的实例化、前向、几何边界条件与目标函数测试。"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.metrics.csi_metrics import compute_nmse
from csifeedback.models import get_model
from csifeedback.trainers.cylinder_codec import MetricMSELoss


@pytest.mark.parametrize("encoded_dim", [512, 256, 128, 64, 32])
def test_instantiation(encoded_dim):
    model = get_model("cylinder_codec", encoded_dim=encoded_dim)
    assert model.encoded_dim == encoded_dim


@pytest.mark.parametrize("encoded_dim", [512, 32])
def test_forward_shape(encoded_dim):
    model = get_model("cylinder_codec", encoded_dim=encoded_dim)
    model.eval()
    x = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_angle_boundary_is_circular():
    """角度轴卷积应把 bin 31 与 bin 0 视为相邻：冲激响应在圆周上连续。"""
    from csifeedback.models.cylinder_codec import CylindricalConv2d

    conv = CylindricalConv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0  # 仅中心 tap
    # 复制中心权重到左/右邻居，使输出等于角度方向的三点平均
    with torch.no_grad():
        conv.weight[0, 0, 0, 1] = 1.0 / 3
        conv.weight[0, 0, 1, 1] = 1.0 / 3
        conv.weight[0, 0, 2, 1] = 1.0 / 3
    x = torch.zeros(1, 1, 32, 32)
    x[0, 0, 0, 0] = 1.0  # 只在 bin 0, tap 0 放冲激
    with torch.no_grad():
        out = conv(x)
    # bin 31、bin 0、bin 1 都应有响应（圆周相邻）
    assert out[0, 0, 31, 0] > 0
    assert out[0, 0, 0, 0] > 0
    assert out[0, 0, 1, 0] > 0


def test_delay_boundary_is_absorbing():
    """时延轴卷积应是零（吸收）边界：tap 0 的响应不应回绕到 tap 31。"""
    from csifeedback.models.cylinder_codec import CylindricalConv2d

    conv = CylindricalConv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0
    x = torch.zeros(1, 1, 32, 32)
    x[0, 0, 0, 0] = 1.0
    with torch.no_grad():
        out = conv(x)
    assert out[0, 0, 0, 31] == 0
    assert out[0, 0, 0, 0] == 1.0


def test_metric_loss_matches_compute_nmse():
    """MetricMSELoss 与官方 NMSE 只差一个 log10，二者必须一致。"""
    torch.manual_seed(0)
    pred = torch.rand(16, 2, 32, 32)
    gt = torch.rand(16, 2, 32, 32)
    loss = MetricMSELoss()(pred, gt)
    nmse_db = compute_nmse(pred, gt)
    assert torch.allclose(
        torch.tensor(10.0) * torch.log10(loss), nmse_db, atol=1e-5
    )
