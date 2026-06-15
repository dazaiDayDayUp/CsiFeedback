"""训练循环的单 batch 过拟合 sanity check。"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from csifeedback.models import get_model


@pytest.mark.parametrize("name,kwargs", [
    ("clnet", {"reduction": 4}),
    ("stnet", {"encoded_dim": 512}),
    ("csinet", {"encoded_dim": 512}),
])
def test_overfit_one_batch(name, kwargs):
    """模型应能过拟合单个 batch。"""
    torch.manual_seed(0)
    model = get_model(name, **kwargs)
    x = torch.randn(4, 2, 32, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    initial_loss = None
    for _ in range(200):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, x)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss * 0.75
