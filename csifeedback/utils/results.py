"""实验结果的元数据收集与落盘。"""

import json
import math
import os
import subprocess
from typing import Any, Dict, Optional


__all__ = ["git_commit", "json_safe", "write_metrics"]


def git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    """返回当前仓库的短 commit hash；不可用时返回 ``None``。"""
    root = repo_root or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def json_safe(value: Any) -> Any:
    """递归地把 NaN/Inf 转成 ``None``，使结果可写入标准 JSON。"""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_metrics(exp_dir: str, payload: Dict[str, Any]) -> str:
    """把运行摘要写入 ``<exp_dir>/metrics.json``，返回文件路径。"""
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return path
