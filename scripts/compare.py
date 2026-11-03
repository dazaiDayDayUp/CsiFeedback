#!/usr/bin/env python
"""汇总并对比实验运行结果。

优先读取每次运行目录下的 ``metrics.json``；对只有 ``logs/train.log`` 的运行，
回退解析其中记录的最优 test rho / NMSE。

示例::

    # 每个 (model, scenario, cr) 只看最新一次运行
    python scripts/compare.py --latest

    # 只看 baseline，按 NMSE 排序并输出 Markdown 表格
    python scripts/compare.py --models clnet stnet csinet --format markdown

    # 导出 CSV 做进一步分析
    python scripts/compare.py --format csv > results.csv
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TEST_RE = re.compile(
    r"test_loss:\s*[\d.eE+-]+\s+rho:\s*([\d.]+)\s+NMSE:\s*(-?[\d.]+)\s*dB"
)
EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
DEFAULT_MIN_EPOCHS = 100


@dataclass
class Run:
    model: str
    scenario: str
    cr: int
    rho: Optional[float]
    nmse: Optional[float]
    epochs: Optional[int]
    selection: str
    run: str
    source: str
    finished_at: str = ""


def _split_group(group: str) -> tuple[str, int]:
    """把 ``indoor_cr4`` 拆成 ``("indoor", 4)``。"""
    match = re.fullmatch(r"(.+)_cr(\d+)", group)
    if not match:
        return group, -1
    return match.group(1), int(match.group(2))


def _select_best(data: dict) -> str:
    """读取运行使用的 checkpoint 选择策略。

    优先顶层 ``select_best``；旧版 ``metrics.json`` 只把它写在
    ``config.training.select_best`` 中（扁平点分键或嵌套字典），因此同时兼容。
    """
    select_best = data.get("select_best")
    if select_best is None:
        config = data.get("config")
        if isinstance(config, dict):
            select_best = config.get("training.select_best")
            if select_best is None:
                training = config.get("training")
                if isinstance(training, dict):
                    select_best = training.get("select_best")
    return str(select_best) if select_best else "unknown"


def _metric_pair(entry: object) -> tuple[Optional[float], Optional[float]]:
    """从 ``{"rho": ..., "nmse": ...}`` 条目中取两个指标。"""
    if not isinstance(entry, dict):
        return None, None
    return entry.get("rho"), entry.get("nmse")


def _selected_metrics(data: dict) -> tuple[Optional[float], Optional[float]]:
    """按 ``select_best`` 取出被选检查点的 rho / NMSE。

    ``selected`` 缺失或缺少对应条目时返回 ``(None, None)``；是否回退到
    ``best`` 由调用方根据选择策略是否已知决定。
    """
    selected = data.get("selected")
    if not isinstance(selected, dict):
        return None, None

    selection = _select_best(data)
    if selection == "val_loss":
        return _metric_pair(selected.get("best_val"))
    if selection == "test_rho":
        return _metric_pair(selected.get("best_rho"))
    if selection == "test_nmse":
        return _metric_pair(selected.get("best_nmse"))
    if selection == "all":
        rho, _ = _metric_pair(selected.get("best_rho"))
        _, nmse = _metric_pair(selected.get("best_nmse"))
        return rho, nmse
    return None, None


def _infer_model(relative: Path, root: Path, fallback: str = "?") -> str:
    """按 ``<model>/<scenario>_cr<X>/<run>`` 的相对路径推断模型名。

    ``root`` 直接指向模型目录时相对路径只剩场景与运行目录，用 ``root.name``
    作为模型名。
    """
    if len(relative.parts) >= 3:
        return relative.parts[-3]
    if len(relative.parts) == 2:
        return root.name
    return fallback


def _from_metrics(path: Path, root: Path) -> Run:
    data = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    relative = run_dir.relative_to(root)
    model = _infer_model(relative, root, fallback=str(data.get("model") or "?"))
    scenario = str(data.get("scenario", "?"))
    cr_label = str(data.get("compression_ratio", "cr?"))
    cr = int(cr_label[2:]) if cr_label.startswith("cr") and cr_label[2:].isdigit() else -1
    best = data.get("best") or {}
    selection = _select_best(data)
    if isinstance(data.get("selected"), dict) and selection != "unknown":
        rho, nmse = _selected_metrics(data)
    else:
        rho, nmse = best.get("rho"), best.get("nmse")
    return Run(
        model=model,
        scenario=scenario,
        cr=cr,
        rho=rho,
        nmse=nmse,
        epochs=data.get("epochs_completed"),
        selection=selection,
        run=run_dir.name,
        source="metrics.json",
        finished_at=str(data.get("finished_at") or ""),
    )


def _from_log(path: Path, root: Path) -> Optional[Run]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = TEST_RE.findall(text)
    if not matches:
        return None

    rhos = [float(rho) for rho, _ in matches]
    nmses = [float(nmse) for _, nmse in matches]
    epochs = [int(done) for done, _ in EPOCH_RE.findall(text)]
    run_dir = path.parent.parent
    relative = run_dir.relative_to(root)
    model = _infer_model(relative, root)
    if len(relative.parts) >= 2:
        scenario, cr = _split_group(relative.parts[-2])
    else:
        scenario, cr = "?", -1

    return Run(
        model=model,
        scenario=scenario,
        cr=cr,
        rho=max(rhos),
        nmse=min(nmses),
        epochs=max(epochs) if epochs else None,
        selection="unknown",
        run=run_dir.name,
        source="train.log",
    )


def discover(root: Path, include_logs: bool = True) -> list[Run]:
    runs: list[Run] = []
    covered: set[Path] = set()

    for metrics_path in root.rglob("metrics.json"):
        runs.append(_from_metrics(metrics_path, root))
        covered.add(metrics_path.parent)

    if include_logs:
        for log_path in root.rglob("train.log"):
            run_dir = log_path.parent.parent
            if run_dir in covered:
                continue
            run = _from_log(log_path, root)
            if run is not None:
                runs.append(run)

    return runs


def filter_runs(
    runs: Iterable[Run],
    models: Optional[list[str]],
    scenarios: Optional[list[str]],
    crs: Optional[list[int]],
) -> list[Run]:
    result = list(runs)
    if models:
        wanted = {m.lower() for m in models}
        result = [r for r in result if r.model.lower() in wanted]
    if scenarios:
        wanted = {s.lower() for s in scenarios}
        result = [r for r in result if r.scenario.lower() in wanted]
    if crs:
        wanted_cr = set(crs)
        result = [r for r in result if r.cr in wanted_cr]
    return result


def filter_min_epochs(runs: Iterable[Run], min_epochs: int) -> list[Run]:
    """过滤完成 epoch 数不足的运行；未知 epoch 按 0 处理。"""
    result = list(runs)
    if min_epochs <= 0:
        return result
    return [run for run in result if (run.epochs or 0) >= min_epochs]


def keep_latest(runs: Iterable[Run]) -> list[Run]:
    latest: dict[tuple[str, str, int], Run] = {}
    for run in runs:
        key = (run.model, run.scenario, run.cr)
        current = latest.get(key)
        if current is None or (run.finished_at, run.run) > (
            current.finished_at,
            current.run,
        ):
            latest[key] = run
    return list(latest.values())


def sort_runs(runs: Iterable[Run], key: str) -> list[Run]:
    if key == "rho":
        return sorted(runs, key=lambda r: (r.rho is None, -(r.rho or 0.0)))
    if key == "model":
        return sorted(runs, key=lambda r: (r.model, r.scenario, r.cr, r.run))
    return sorted(runs, key=lambda r: (r.nmse is None, r.nmse if r.nmse is not None else 0.0))


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _rows(runs: Iterable[Run]) -> list[list[str]]:
    return [
        [
            run.model,
            run.scenario,
            f"1/{run.cr}" if run.cr > 0 else "—",
            _fmt(run.rho),
            _fmt(run.nmse),
            str(run.epochs) if run.epochs is not None else "—",
            run.selection,
            run.source,
            run.run,
        ]
        for run in runs
    ]


HEADER = [
    "model",
    "scenario",
    "CR",
    "rho",
    "NMSE(dB)",
    "epochs",
    "select",
    "source",
    "run",
]


def _print_table(rows: list[list[str]]) -> None:
    widths = [len(h) for h in HEADER]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print(line(HEADER))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def _print_markdown(rows: list[list[str]]) -> None:
    print("| " + " | ".join(HEADER) + " |")
    print("|" + "|".join("---" for _ in HEADER) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总并对比 CSI 反馈实验运行")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments"),
        help="实验根目录，默认 experiments",
    )
    parser.add_argument("--models", nargs="+", help="只显示指定模型")
    parser.add_argument("--scenarios", nargs="+", help="只显示指定场景（indoor/outdoor）")
    parser.add_argument("--cr", nargs="+", type=int, help="只显示指定压缩比（4/8/16/32/64）")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="每个 (model, scenario, cr) 只保留最新一次运行",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="只读取 metrics.json，不解析只有 train.log 的运行",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
        help="只显示至少完成 N 个 epoch 的运行（默认 100，过滤 smoke test 等短运行）",
    )
    parser.add_argument(
        "--sort",
        choices=["nmse", "rho", "model"],
        default="nmse",
        help="排序方式，默认按 NMSE 从优到劣",
    )
    parser.add_argument(
        "--format",
        choices=["table", "markdown", "csv"],
        default="table",
        help="输出格式",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"实验目录不存在：{root}")

    runs = discover(root, include_logs=not args.metrics_only)
    runs = filter_runs(runs, args.models, args.scenarios, args.cr)
    runs = filter_min_epochs(runs, args.min_epochs)
    if args.latest:
        runs = keep_latest(runs)
    runs = sort_runs(runs, args.sort)

    rows = _rows(runs)
    if args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(HEADER)
        writer.writerows(rows)
    elif args.format == "markdown":
        _print_markdown(rows)
    else:
        _print_table(rows)
        print(f"\n共 {len(rows)} 次运行（root={root}）")


if __name__ == "__main__":
    main()
