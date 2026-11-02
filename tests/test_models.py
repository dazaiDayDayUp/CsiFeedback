"""模型实例化、前向形状与 baseline 行为稳定性的测试。"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.models import get_model


@pytest.mark.parametrize("reduction", [4, 8, 16, 32, 64])
def test_clnet_instantiation(reduction):
    model = get_model("clnet", reduction=reduction)
    assert model.encoded_dim == 2048 // reduction


@pytest.mark.parametrize("encoded_dim", [512, 256, 128, 64, 32])
def test_stnet_instantiation(encoded_dim):
    model = get_model("stnet", encoded_dim=encoded_dim)
    assert model.encoded_dim == encoded_dim


@pytest.mark.parametrize("encoded_dim", [512, 256, 128, 64, 32])
def test_csinet_instantiation(encoded_dim):
    model = get_model("csinet", encoded_dim=encoded_dim)
    assert model.encoded_dim == encoded_dim


@pytest.mark.parametrize("name,kwargs", [
    ("clnet", {"reduction": 4}),
    ("stnet", {"encoded_dim": 512}),
    ("csinet", {"encoded_dim": 512}),
])
def test_forward_shape(name, kwargs):
    model = get_model(name, **kwargs)
    model.eval()
    x = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("name,kwargs", [
    ("clnet", {"reduction": 4}),
    ("stnet", {"encoded_dim": 512}),
    ("csinet", {"encoded_dim": 512}),
])
def test_deterministic_forward(name, kwargs):
    torch.manual_seed(0)
    model = get_model(name, **kwargs)
    model.eval()
    x = torch.randn(1, 2, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_clnet_encoder_decoder_separation():
    model = get_model("clnet", reduction=4)
    enc_state = model.get_encoder_state()
    dec_state = model.get_decoder_state()
    assert len(enc_state) > 0
    assert len(dec_state) > 0
    assert not set(enc_state.keys()) & set(dec_state.keys())


def test_model_config_accepts_unknown_name_and_extra():
    """ModelConfig 不再硬编码校验模型名，并支持 extra 参数字段。"""
    from csifeedback.utils.config import ModelConfig
    from csifeedback.models import _MODEL_FACTORIES, CSIAutoencoder, get_model_from_config

    # 临时注册一个 mock 工厂
    def create_mocknet(encoded_dim: int = 512, depth: int = 1, **kwargs):
        class MockNet(CSIAutoencoder):
            def __init__(self, encoded_dim, depth):
                super().__init__()
                self._encoded_dim = encoded_dim
                self.depth = depth

            @property
            def encoded_dim(self):
                return self._encoded_dim

            def get_encoder_state(self):
                return {}

            def get_decoder_state(self):
                return {}

            def forward(self, x):
                return x

        return MockNet(encoded_dim, depth)

    _MODEL_FACTORIES["mocknet"] = create_mocknet
    try:
        config = ModelConfig(name="mocknet", encoded_dim=256, extra={"depth": 3})
        model = get_model_from_config(config)
        assert model.encoded_dim == 256
        assert model.depth == 3
    finally:
        del _MODEL_FACTORIES["mocknet"]


def test_model_factory_unknown_name_gives_helpful_error():
    from csifeedback.utils.config import ModelConfig
    from csifeedback.models import get_model_from_config

    config = ModelConfig(name="notregistered")
    with pytest.raises(ValueError, match="请在 csifeedback/models/__init__.py"):
        get_model_from_config(config)


def test_stnet_decoder_out_broadcast_preserved():
    """确保 decoder 仍以复用 out 的方式运行两个 WTL 循环。"""
    model = get_model("stnet", encoded_dim=512)
    x = torch.randn(1, 2, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 32, 32)
    model = get_model("stnet", encoded_dim=512)
    x = torch.randn(1, 2, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 32, 32)
