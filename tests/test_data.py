"""统一 COST2100 数据加载器的测试。"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.data.cost2100 import Cost2100Dataset, get_cost2100_loaders


DATA_DIR = repo_root / "data"


def _data_available():
    return (DATA_DIR / "DATA_Htrainin.mat").is_file()


@pytest.mark.skipif(not _data_available(), reason="data/ 中未找到 COST2100 数据")
def test_dataset_train_shape():
    ds = Cost2100Dataset(str(DATA_DIR), "indoor", "train")
    assert len(ds) > 0
    sample = ds[0]
    assert sample[0].shape == (2, 32, 32)


@pytest.mark.skipif(not _data_available(), reason="data/ 中未找到 COST2100 数据")
def test_dataset_test_with_raw():
    ds = Cost2100Dataset(str(DATA_DIR), "in", "test", load_raw=True)
    sample = ds[0]
    assert len(sample) == 2
    sparse, raw = sample
    assert sparse.shape == (2, 32, 32)
    assert raw.shape == (32, 125, 2)


@pytest.mark.skipif(not _data_available(), reason="data/ 中未找到 COST2100 数据")
def test_loaders():
    train_loader, val_loader, test_loader = get_cost2100_loaders(
        str(DATA_DIR), "indoor", batch_size=16, load_raw=True
    )
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    batch = next(iter(test_loader))
    assert len(batch) == 2
    sparse, raw = batch
    assert sparse.shape[1:] == (2, 32, 32)
    assert raw.shape[1:] == (32, 125, 2)


@pytest.mark.skipif(not _data_available(), reason="data/ 中未找到 COST2100 数据")
def test_scenario_alias():
    ds_indoor = Cost2100Dataset(str(DATA_DIR), "indoor", "train")
    ds_in = Cost2100Dataset(str(DATA_DIR), "in", "train")
    assert len(ds_indoor) == len(ds_in)
