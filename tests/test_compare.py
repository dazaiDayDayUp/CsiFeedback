"""``scripts/compare.py`` 的运行解析、选择策略与过滤逻辑测试。"""

import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.compare import (
    DEFAULT_MIN_EPOCHS,
    HEADER,
    Run,
    _from_log,
    _from_metrics,
    filter_min_epochs,
)


SELECTED = {
    "best_val": {"rho": 0.71, "nmse": -15.0},
    "best_rho": {"rho": 0.82, "nmse": -16.0},
    "best_nmse": {"rho": 0.80, "nmse": -17.0},
}
BEST = {"rho": 0.90, "nmse": -18.0}


def _metrics_payload(select_best: str = "all", **overrides) -> dict:
    payload = {
        "scenario": "indoor",
        "compression_ratio": "cr4",
        "epochs_completed": 1000,
        "finished_at": "2026-01-01T00:00:00",
        "best": dict(BEST),
        "selected": {key: dict(value) for key, value in SELECTED.items()},
        "config": {"training": {"select_best": select_best}},
    }
    payload.update(overrides)
    return payload


def _write_metrics(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("select_best", "expected_rho", "expected_nmse"),
    [
        ("val_loss", 0.71, -15.0),
        ("test_rho", 0.82, -16.0),
        ("test_nmse", 0.80, -17.0),
        ("all", 0.82, -17.0),
    ],
)
def test_from_metrics_uses_selected_strategy(
    tmp_path: Path,
    select_best: str,
    expected_rho: float,
    expected_nmse: float,
):
    path = _write_metrics(
        tmp_path / "clnet" / "indoor_cr4" / "run1",
        _metrics_payload(select_best),
    )
    run = _from_metrics(path, tmp_path)

    assert run.selection == select_best
    assert run.rho == pytest.approx(expected_rho)
    assert run.nmse == pytest.approx(expected_nmse)


def test_from_metrics_reads_top_level_select_best(tmp_path: Path):
    payload = _metrics_payload("all")
    payload["select_best"] = "test_rho"
    path = _write_metrics(tmp_path / "clnet" / "indoor_cr4" / "run1", payload)
    run = _from_metrics(path, tmp_path)

    assert run.selection == "test_rho"
    assert run.rho == pytest.approx(SELECTED["best_rho"]["rho"])
    assert run.nmse == pytest.approx(SELECTED["best_rho"]["nmse"])


def test_from_metrics_reads_flat_config_select_best(tmp_path: Path):
    payload = _metrics_payload("all")
    payload["config"] = {"training.select_best": "test_nmse"}
    path = _write_metrics(tmp_path / "clnet" / "indoor_cr4" / "run1", payload)
    run = _from_metrics(path, tmp_path)

    assert run.selection == "test_nmse"
    assert run.rho == pytest.approx(SELECTED["best_nmse"]["rho"])
    assert run.nmse == pytest.approx(SELECTED["best_nmse"]["nmse"])


def test_from_metrics_falls_back_without_selected(tmp_path: Path):
    payload = _metrics_payload("val_loss")
    del payload["selected"]
    path = _write_metrics(tmp_path / "clnet" / "indoor_cr4" / "run1", payload)
    run = _from_metrics(path, tmp_path)

    assert run.selection == "val_loss"
    assert run.rho == pytest.approx(BEST["rho"])
    assert run.nmse == pytest.approx(BEST["nmse"])


def test_from_metrics_keeps_missing_selected_metric_empty(tmp_path: Path):
    payload = _metrics_payload("val_loss")
    payload["selected"] = {"best_val": {"epoch": 3, "val_loss": 1e-5}}
    path = _write_metrics(tmp_path / "clnet" / "indoor_cr4" / "run1", payload)
    run = _from_metrics(path, tmp_path)

    assert run.rho is None
    assert run.nmse is None


def test_from_metrics_marks_unknown_without_select_best(tmp_path: Path):
    payload = _metrics_payload("all")
    del payload["config"]
    path = _write_metrics(tmp_path / "clnet" / "indoor_cr4" / "run1", payload)
    run = _from_metrics(path, tmp_path)

    assert run.selection == "unknown"
    assert run.rho == pytest.approx(BEST["rho"])
    assert run.nmse == pytest.approx(BEST["nmse"])


def test_from_metrics_infers_model_when_root_is_model_dir(tmp_path: Path):
    root = tmp_path / "clnet"
    path = _write_metrics(root / "indoor_cr4" / "run1", _metrics_payload())
    run = _from_metrics(path, root)

    assert run.model == "clnet"
    assert run.scenario == "indoor"
    assert run.cr == 4


@pytest.mark.parametrize("root_is_model", [False, True])
def test_from_log_infers_model(tmp_path: Path, root_is_model: bool):
    if root_is_model:
        root = tmp_path / "clnet"
        run_dir = root / "indoor_cr4" / "run1"
    else:
        root = tmp_path
        run_dir = root / "clnet" / "indoor_cr4" / "run1"
    log_path = run_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "Epoch [20/20] test_loss: 1.0000e-02  rho: 0.8123  NMSE: -14.5000 dB\n",
        encoding="utf-8",
    )

    run = _from_log(log_path, root)

    assert run is not None
    assert run.model == "clnet"
    assert run.scenario == "indoor"
    assert run.cr == 4
    assert run.epochs == 20
    assert run.selection == "unknown"


def _make_run(epochs, name: str) -> Run:
    return Run(
        model="clnet",
        scenario="indoor",
        cr=4,
        rho=0.8,
        nmse=-15.0,
        epochs=epochs,
        selection="all",
        run=name,
        source="metrics.json",
    )


def test_filter_min_epochs_default_drops_smoke_runs():
    runs = [
        _make_run(20, "smoke"),
        _make_run(99, "short"),
        _make_run(100, "ok"),
        _make_run(1000, "full"),
        _make_run(None, "unknown"),
    ]

    filtered = filter_min_epochs(runs, DEFAULT_MIN_EPOCHS)

    assert DEFAULT_MIN_EPOCHS == 100
    assert [run.run for run in filtered] == ["ok", "full"]


def test_header_places_select_after_epochs():
    assert HEADER.index("select") == HEADER.index("epochs") + 1
    assert HEADER.index("source") == HEADER.index("select") + 1
