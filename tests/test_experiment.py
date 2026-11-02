"""模型注册、通用 trainer 与运行摘要的测试。"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.models import (
    _MODEL_FACTORIES,
    CSIAutoencoder,
    available_models,
    get_model,
    register_model,
)
from csifeedback.trainers import DefaultTrainer, get_trainer
from csifeedback.utils.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


class TinyModel(CSIAutoencoder):
    """最小的自编码器，用于测试扩展点。"""

    def __init__(self, encoded_dim: int = 8):
        super().__init__()
        self._encoded_dim = encoded_dim
        self.scale = nn.Parameter(torch.tensor(0.9))

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> dict:
        return {}

    def get_decoder_state(self) -> dict:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _make_config(
    tmp_path: Path,
    name: str,
    scheduler: str = "const",
    select_best: str = "all",
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(name=name, encoded_dim=256),
        data=DataConfig(scenario="indoor", batch_size=2, load_raw=False),
        optimizer=OptimizerConfig(name="adam", lr=1e-3),
        scheduler=SchedulerConfig(name=scheduler),
        training=TrainingConfig(
            epochs=2,
            device="cpu",
            seed=0,
            val_freq=1,
            test_freq=1,
            print_freq=0,
            checkpoint_dir=str(tmp_path),
            checkpoint_freq=0,
            select_best=select_best,
        ),
    )


def test_register_model_decorator():
    @register_model("tiny_test_net")
    def create_tiny(encoded_dim: int = 8, **kwargs) -> TinyModel:
        return TinyModel(encoded_dim=encoded_dim)

    try:
        assert "tiny_test_net" in available_models()
        model = get_model("tiny_test_net", encoded_dim=16)
        assert model.encoded_dim == 16
    finally:
        _MODEL_FACTORIES.pop("tiny_test_net", None)


def test_default_trainer_writes_metrics(tmp_path):
    config = _make_config(tmp_path, "unregistered_tiny_net")
    model = TinyModel()
    trainer = get_trainer(config, model, torch.device("cpu"))
    assert isinstance(trainer, DefaultTrainer)

    data = torch.rand(4, 2, 32, 32)
    loader = DataLoader(TensorDataset(data), batch_size=2)
    trainer.fit(loader, loader, loader)

    metrics_path = Path(trainer.exp_dir) / "metrics.json"
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"] == "unregistered_tiny_net"
    assert metrics["compression_ratio"] == "cr8"  # 2048 / encoded_dim=256
    assert metrics["epochs_completed"] == 2
    assert metrics["best"]["nmse"] is not None
    assert len(metrics["history"]) == 4  # 2 epochs × (val + test)
    assert metrics["config"]["training.epochs"] == 2


def test_val_based_checkpoint_selection(tmp_path):
    config = _make_config(tmp_path, "unregistered_tiny_net", select_best="val_loss")
    trainer = DefaultTrainer(config, TinyModel(), torch.device("cpu"))

    data = torch.rand(4, 2, 32, 32)
    loader = DataLoader(TensorDataset(data), batch_size=2)
    trainer.fit(loader, loader, loader)

    checkpoint_dir = Path(trainer.exp_dir) / "checkpoints"
    assert (checkpoint_dir / "best_val.pth").is_file()
    assert not (checkpoint_dir / "best_rho.pth").exists()
    assert not (checkpoint_dir / "best_nmse.pth").exists()


def test_default_trainer_prepares_cosine_schedule(tmp_path):
    config = _make_config(tmp_path, "unregistered_tiny_net", scheduler="cosine")
    trainer = DefaultTrainer(config, TinyModel(), torch.device("cpu"))

    data = torch.rand(4, 2, 32, 32)
    loader = DataLoader(TensorDataset(data), batch_size=2)
    trainer._prepare_scheduler(loader)

    total_steps = config.training.epochs * len(loader)
    assert trainer.scheduler.T_max == total_steps
    assert trainer.scheduler.T_warmup == min(30 * len(loader), total_steps - 1)
