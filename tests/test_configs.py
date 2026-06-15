"""YAML 配置矩阵的加载与实例化测试。

遍历 csifeedback/configs/ 下所有 YAML 配置文件，确保每个配置都能被
正确加载、实例化对应模型，并在随机输入上完成一次前向传播。
"""

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.models import get_model_from_config
from csifeedback.utils.config import load_config


CONFIG_DIR = repo_root / "csifeedback" / "configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=[p.name for p in CONFIG_PATHS])
def test_config_loads_and_model_forward(config_path):
    """每个 YAML 配置都应能加载并驱动对应模型完成一次前向传播。"""
    config = load_config(str(config_path))
    model = get_model_from_config(config.model)

    x = torch.randn(2, 2, 32, 32)
    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert model.encoded_dim > 0
