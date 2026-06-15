#!/usr/bin/env python
"""所有 CSI 反馈模型的统一评估入口。"""

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch

from csifeedback.data.cost2100 import get_cost2100_loaders
from csifeedback.metrics.csi_metrics import compute_nmse, compute_rho
from csifeedback.models import get_model_from_config
from csifeedback.utils.checkpoint import load_checkpoint
from csifeedback.utils.config import load_config
from csifeedback.utils.device import setup_device


def main():
    parser = argparse.ArgumentParser(description="评估 CSI 反馈模型")
    parser.add_argument("--config", "-c", required=True, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="检查点路径")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="要评估的数据集划分",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = setup_device(config.training.device)

    _, val_loader, test_loader = get_cost2100_loaders(
        data_dir=config.data.data_dir,
        scenario=config.data.scenario,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        load_raw=True,
    )
    loader = {"train": None, "val": val_loader, "test": test_loader}[args.split]
    if loader is None:
        raise ValueError("统一加载器不支持在 train 划分上评估")

    model = get_model_from_config(config.model).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(ckpt.model_state_dict)
    model.eval()

    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    n_batches = 0
    all_pred = []
    all_gt = []
    all_raw = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, raw = batch
                all_raw.append(raw)
            else:
                x = batch[0]
            x = x.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, x)
            total_loss += loss.item()
            n_batches += 1
            all_pred.append(out.cpu())
            all_gt.append(x.cpu())

    preds = torch.cat(all_pred, dim=0)
    gts = torch.cat(all_gt, dim=0)
    nmse = compute_nmse(preds, gts).item()

    if all_raw:
        raw = torch.cat(all_raw, dim=0)
        rho = compute_rho(preds, raw).item()
    else:
        rho = float("nan")

    avg_loss = total_loss / max(1, n_batches)
    print(f"Split: {args.split}")
    print(f"Loss:  {avg_loss:.4e}")
    print(f"NMSE:  {nmse:.4f} dB")
    print(f"Rho:   {rho:.6f}")


if __name__ == "__main__":
    main()
