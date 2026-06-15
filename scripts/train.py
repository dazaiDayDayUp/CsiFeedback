#!/usr/bin/env python
"""所有 CSI 反馈模型的统一训练入口。"""

import argparse
import sys
from pathlib import Path

# 允许在未安装包的情况下直接从仓库根目录运行。
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.data.cost2100 import get_cost2100_loaders
from csifeedback.models import get_model_from_config
from csifeedback.trainers import get_trainer
from csifeedback.utils.config import load_config
from csifeedback.utils.device import set_seed, setup_device
from csifeedback.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="训练 CSI 反馈模型")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="YAML 配置文件路径（例如 csifeedback/configs/clnet_indoor_cr4.yaml）",
    )
    parser.add_argument(
        "--override", "-o",
        nargs="*",
        default=[],
        help="覆盖配置项，例如：-o training.epochs=50 data.batch_size=100",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)

    # 先配置日志，确保训练器的日志能输出到控制台。
    setup_logging()

    device = setup_device(config.training.device)
    set_seed(config.training.seed)

    train_loader, val_loader, test_loader = get_cost2100_loaders(
        data_dir=config.data.data_dir,
        scenario=config.data.scenario,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        load_raw=config.data.load_raw,
    )

    model = get_model_from_config(config.model).to(device)
    trainer = get_trainer(config, model, device)
    trainer.fit(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
