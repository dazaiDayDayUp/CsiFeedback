#!/usr/bin/env python
"""批量训练包装脚本：按顺序调用 scripts/train.py 训练所有 YAML 配置。

用法示例::

    python scripts/train_all.py
    python scripts/train_all.py --pattern "clnet_*.yaml"

每个配置的训练设备由对应 YAML 文件中的 ``training.device`` 决定，
与 ``scripts/train.py`` 的行为保持一致。
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "csifeedback" / "configs"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def main():
    parser = argparse.ArgumentParser(description="批量训练所有 CSI 反馈配置")
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="配置文件匹配模式（默认：*.yaml）",
    )
    args = parser.parse_args()

    configs = sorted(
        CONFIG_DIR.glob(args.pattern),
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)],
    )
    if not configs:
        print(f"未找到匹配 {args.pattern!r} 的配置文件：{CONFIG_DIR}")
        return

    for cfg in configs:
        print(f"\n>>> [{cfg.name}] 开始训练")
        subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--config", str(cfg)],
            check=False,
        )


if __name__ == "__main__":
    main()
