"""配置系统：dataclass + YAML，支持命令行覆盖。"""

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


__all__ = [
    "ModelConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "load_config",
    "override_config",
]


@dataclass
class ModelConfig:
    name: str = "clnet"
    reduction: Optional[int] = None       # CLNet 使用
    encoded_dim: Optional[int] = None     # STNet / CsiNet 使用
    extra: Dict[str, Any] = field(default_factory=dict)  # 新模型自定义参数

    def __post_init__(self) -> None:
        self.name = self.name.lower()
        if not self.name:
            raise ValueError("model.name 不能为空")


@dataclass
class DataConfig:
    data_dir: str = "../data"
    scenario: str = "indoor"  # indoor/outdoor，也接受 in/out
    batch_size: int = 200
    num_workers: int = 0
    pin_memory: bool = False
    load_raw: bool = True  # 测试集是否需要加载原始 CSI 以计算 rho


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    separate_enc_dec: bool = False  # STNet 需要分别优化 encoder/decoder

    def __post_init__(self) -> None:
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)


@dataclass
class SchedulerConfig:
    name: str = "const"  # const 或 cosine
    T_max: Optional[int] = None      # 总步数；None 表示自动计算
    T_warmup: Optional[int] = None   # warmup 步数；None 表示自动计算（30 个 epoch）
    eta_min: float = 5e-5


@dataclass
class TrainingConfig:
    epochs: int = 1000
    device: Optional[str] = "auto"
    seed: Optional[int] = None
    val_freq: int = 10
    test_freq: int = 10
    print_freq: int = 20
    checkpoint_dir: str = "./experiments"
    checkpoint_freq: int = 50  # 每 N 个 epoch 保存一次完整检查点，0 表示关闭
    resume: Optional[str] = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _flatten_dataclass(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """将嵌套 dataclass 展平为点分键字典。"""
    result: Dict[str, Any] = {}
    for f in fields(obj):
        key = f"{prefix}{f.name}" if not prefix else f"{prefix}.{f.name}"
        value = getattr(obj, f.name)
        if hasattr(value, "__dataclass_fields__"):
            result.update(_flatten_dataclass(value, key))
        else:
            result[key] = value
    return result


def _set_nested(config: Any, key: str, value: Any) -> None:
    """在嵌套 dataclass 上设置点分属性。"""
    parts = key.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    field_type = type(getattr(obj, parts[-1]))
    try:
        # 尽量保持原始类型（如 int、float、tuple）。
        if field_type is tuple:
            converted = tuple(
                float(x) if "." in x else int(x) for x in str(value).strip("()[]").split(",")
            )
        elif field_type is bool:
            converted = str(value).lower() in {"true", "1", "yes"}
        else:
            converted = field_type(value)
    except (ValueError, TypeError):
        converted = value
    setattr(obj, parts[-1], converted)


def load_config(path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """加载 YAML 配置，并可选择性地应用命令行覆盖（key=value）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = ExperimentConfig(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )

    if overrides:
        config = override_config(config, overrides)
    return config


def override_config(config: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """对配置副本应用 ``key=value`` 形式的覆盖。"""
    config = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项必须是 key=value 格式：{item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        # 支持 betas 等字段的逗号分隔列表。
        if "," in value and not value.startswith("["):
            value = [v.strip() for v in value.split(",")]
        _set_nested(config, key, value)
    return config


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """将配置序列化为普通字典（适用于 YAML/检查点）。"""
    return _flatten_dataclass(config)
