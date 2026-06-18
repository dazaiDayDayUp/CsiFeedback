#!/usr/bin/env python
"""批量训练 csifeedback/configs/ 下的所有 YAML 配置。"""

import argparse
import fnmatch
import re
import subprocess
import sys
import time
from pathlib import Path


def _cr_sort_key(path: Path) -> tuple[int | float, str]:
    """按文件名中的压缩比 crX 数字升序排序；无法解析时按文件名排序。"""
    match = re.search(r"cr(\d+)", path.name, flags=re.IGNORECASE)
    cr = int(match.group(1)) if match else float("inf")
    return (cr, path.name)


def find_configs(config_dir: Path, pattern: str) -> list[Path]:
    """根据通配符模式返回匹配的配置文件列表，按压缩比升序、再按文件名排序。"""
    return sorted(
        (p for p in config_dir.glob("*.yaml") if fnmatch.fnmatch(p.name, pattern)),
        key=_cr_sort_key,
    )


def format_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS。"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(description="批量训练所有 CSI 反馈模型配置")
    parser.add_argument(
        "--pattern", "-p",
        default="*.yaml",
        help="配置文件匹配模式（例如 clnet_*.yaml），默认 *.yaml",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "csifeedback" / "configs"
    train_script = repo_root / "scripts" / "train.py"

    configs = find_configs(config_dir, args.pattern)
    if not configs:
        print(f"未在 {config_dir} 下找到匹配 {args.pattern!r} 的配置文件。", file=sys.stderr)
        sys.exit(1)

    total_start = time.perf_counter()
    print(f"共找到 {len(configs)} 个待训练配置，开始批量训练...\n")

    for idx, config_path in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] 开始训练: {config_path}")
        cmd = [sys.executable, str(train_script), "--config", str(config_path)]

        step_start = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(repo_root))
        step_elapsed = time.perf_counter() - step_start

        if result.returncode != 0:
            print(f"训练失败: {config_path}（用时 {format_duration(step_elapsed)}）", file=sys.stderr)
            print(f"累计用时: {format_duration(time.perf_counter() - total_start)}", file=sys.stderr)
            sys.exit(result.returncode)

        print(f"完成: {config_path}（本配置用时 {format_duration(step_elapsed)}）")
        print(f"累计用时: {format_duration(time.perf_counter() - total_start)}\n")

    total_elapsed = time.perf_counter() - total_start
    print(f"所有配置训练完成，累计用时: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
