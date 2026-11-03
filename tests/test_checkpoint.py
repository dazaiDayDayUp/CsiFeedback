"""检查点保存与模型状态加载的测试。"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.utils.checkpoint import (
    load_checkpoint,
    load_model_state_dict,
    save_checkpoint,
)
from csifeedback.utils.config import ExperimentConfig


class TinyNestedModel(nn.Module):
    """带独立 encoder/decoder 的最小模型。"""

    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.encoder = nn.Linear(in_dim, 2)
        self.decoder = nn.Linear(2, in_dim)

    def load_encoder_state(self, state_dict: dict, strict: bool = True) -> None:
        self.encoder.load_state_dict(state_dict, strict=strict)

    def load_decoder_state(self, state_dict: dict, strict: bool = True) -> None:
        self.decoder.load_state_dict(state_dict, strict=strict)


def test_load_model_state_dict_flat():
    source = nn.Linear(4, 3)
    target = nn.Linear(4, 3)
    with torch.no_grad():
        source.weight.fill_(0.5)
        source.bias.fill_(0.25)

    load_model_state_dict(target, source.state_dict())

    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)


def test_load_model_state_dict_nested():
    source = TinyNestedModel()
    target = TinyNestedModel()
    with torch.no_grad():
        source.encoder.weight.fill_(0.3)
        source.encoder.bias.fill_(0.1)
        source.decoder.weight.fill_(-0.2)
        source.decoder.bias.fill_(0.4)

    state_dict = {
        "encoder": source.encoder.state_dict(),
        "decoder": source.decoder.state_dict(),
    }
    load_model_state_dict(target, state_dict)

    assert torch.allclose(target.encoder.weight, source.encoder.weight)
    assert torch.allclose(target.encoder.bias, source.encoder.bias)
    assert torch.allclose(target.decoder.weight, source.decoder.weight)
    assert torch.allclose(target.decoder.bias, source.decoder.bias)


def test_save_checkpoint_without_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(2, 2)

    save_checkpoint(
        path="checkpoint.pth",
        epoch=3,
        model_name="tiny",
        config=ExperimentConfig(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
    )

    assert (tmp_path / "checkpoint.pth").is_file()
    ckpt = load_checkpoint("checkpoint.pth", map_location="cpu")
    assert ckpt.epoch == 3
    assert ckpt.model_name == "tiny"
