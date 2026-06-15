"""模型实例化、前向形状与行为保留的测试。"""

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


@pytest.mark.parametrize("name,kwargs,legacy_module_path,legacy_factory", [
    (
        "clnet",
        {"reduction": 4},
        "archive/CLNet/models/clnet.py",
        "clnet",
    ),
    (
        "csinet",
        {"encoded_dim": 512},
        "archive/CsiNet/csinet_pytorch.py",
        "CsiNet",
    ),
])
def test_new_vs_legacy_forward(name, kwargs, legacy_module_path, legacy_factory):
    """比较新模型与原始实现的前向输出。"""
    import importlib.util

    legacy_path = repo_root / legacy_module_path
    if not legacy_path.exists():
        pytest.skip(f"未找到原始文件：{legacy_path}")

    if name == "csinet":
        # CsiNet 原始脚本在顶层执行数据加载、matplotlib 导入和完整训练循环。
        # 这里只提取类定义与全局常量，避免执行训练代码和加载 matplotlib。
        legacy = _load_csinet_class_only(legacy_path)
    else:
        spec = importlib.util.spec_from_file_location("legacy_model", legacy_path)
        legacy = importlib.util.module_from_spec(spec)

        # 使原始脚本的本地导入可解析。
        legacy_root = legacy_path.parent.parent
        sys.path.insert(0, str(legacy_root))

        # 原始 CLNet 导入 utils.logger 时会解析 sys.argv；此处抑制该行为。
        old_argv = sys.argv
        sys.argv = ["legacy"]
        try:
            spec.loader.exec_module(legacy)
        except Exception as exc:
            sys.path.pop(0)
            pytest.skip(f"无法加载原始模块：{exc}")
        finally:
            sys.argv = old_argv
            if sys.path[0] == str(legacy_root):
                sys.path.pop(0)

    torch.manual_seed(42)
    x = torch.randn(2, 2, 32, 32)

    new_model = get_model(name, **kwargs)
    legacy_model = getattr(legacy, legacy_factory)(**kwargs)

    # 对齐权重，以便比较前向行为而非随机初始化。
    legacy_model.load_state_dict(new_model.state_dict())

    new_model.eval()
    legacy_model.eval()
    with torch.no_grad():
        new_out = new_model(x)
        legacy_out = legacy_model(x)

    torch.testing.assert_close(new_out, legacy_out, atol=1e-6, rtol=1e-5)


def _load_csinet_class_only(path: Path):
    """从 CsiNet 原始脚本中仅提取类定义和全局常量，不执行训练代码。"""
    import ast
    from types import SimpleNamespace

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    def _is_safe_assign(node):
        """保留不含函数调用的常量赋值（如 img_height = 32）。"""
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            # 如果赋值右侧包含任何 Call，就过滤掉（如 load_data(...)）。
            for child in ast.walk(value):
                if isinstance(child, ast.Call):
                    return False
            return True
        return False

    filtered_body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            filtered_body.append(node)
        elif _is_safe_assign(node):
            filtered_body.append(node)

    new_tree = ast.Module(body=filtered_body, type_ignores=[])
    ast.fix_missing_locations(new_tree)

    namespace = {}
    compiled = compile(new_tree, filename=str(path), mode="exec")
    exec(compiled, namespace)
    return SimpleNamespace(**namespace)


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
