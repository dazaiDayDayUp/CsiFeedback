"""AxisCodec 的实例化、前向与角度周期算子性质测试。"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.models import get_model
from csifeedback.models.axis_codec import AxisBlock, CircularMix, DenseMix


@pytest.mark.parametrize("encoded_dim", [512, 256, 128, 64, 32])
def test_instantiation(encoded_dim):
    model = get_model("axis_codec", encoded_dim=encoded_dim)
    assert model.encoded_dim == encoded_dim


@pytest.mark.parametrize("encoded_dim", [512, 32])
def test_forward_shape(encoded_dim):
    model = get_model("axis_codec", encoded_dim=encoded_dim)
    model.eval()
    x = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_circular_mix_is_shift_equivariant():
    """角度轴循环卷积必须与角度平移严格对易（圆群等变）。"""
    torch.manual_seed(0)
    mix = CircularMix(channels=4)
    x = torch.randn(2, 4, 32, 32)
    for shift in (1, 5, 17):
        lhs = mix(torch.roll(x, shift, dims=2))
        rhs = torch.roll(mix(x), shift, dims=2)
        assert torch.allclose(lhs, rhs, atol=1e-5)


def test_circular_mix_init_is_identity():
    """恒等初始化：频域权重全 1 时输出等于输入。"""
    mix = CircularMix(channels=3)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(mix(x), x, atol=1e-5)


def test_dense_mix_axis_direction():
    """DenseMix(axis=2) 沿角度轴混合，DenseMix(axis=3) 沿时延轴混合。"""
    torch.manual_seed(0)
    m2 = DenseMix(32, axis=2)
    m3 = DenseMix(32, axis=3)
    with torch.no_grad():  # 去掉偏置，否则常数项会掩盖稀疏响应
        m2.fc.bias.zero_()
        m3.fc.bias.zero_()
    # 单点冲激的响应范围必须只在被混合的那条轴上：
    # 轴 3（时延）混合器对 (angle=5, delay=7) 的冲激只能在 angle=5 这一行非零；
    # 轴 2（角度）混合器只能在 delay=7 这一列非零。
    x = torch.zeros(1, 1, 32, 32)
    x[0, 0, 5, 7] = 1.0
    with torch.no_grad():
        y2 = m2(x)
        y3 = m3(x)
    rows = y3[0, 0].abs().sum(dim=1)
    cols = y2[0, 0].abs().sum(dim=0)
    assert rows[torch.arange(32) != 5].max() < 1e-6
    assert rows[5] > 1e-3
    assert cols[torch.arange(32) != 7].max() < 1e-6
    assert cols[7] > 1e-3


def test_block_keeps_shape():
    blk = AxisBlock(8)
    x = torch.randn(2, 8, 32, 32)
    assert blk(x).shape == x.shape


def test_bypass_flag_changes_output_path():
    """线性旁路开关必须真的改变前向路径（消融有效性）。"""
    torch.manual_seed(0)
    a = get_model("axis_codec", encoded_dim=512, bypass=True)
    b = get_model("axis_codec", encoded_dim=512, bypass=False)
    b.load_state_dict(a.state_dict(), strict=False)
    x = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    assert not torch.allclose(ya, yb)
    assert ya.shape == yb.shape == x.shape
