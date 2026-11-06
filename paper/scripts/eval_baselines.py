"""固定 checkpoint 复评三个 baseline（indoor，5 个 CR），用于论文 Table II。

NMSE 取 best_nmse.pth，rho 取 best_rho.pth，与 baseline 的 select_best="all" 协议一致。
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from csifeedback.data.cost2100 import get_cost2100_loaders
from csifeedback.metrics.csi_metrics import compute_nmse, compute_rho
from csifeedback.models import get_model_from_config
from csifeedback.utils.checkpoint import load_checkpoint, load_model_state_dict
from csifeedback.utils.config import load_config

MAIN = Path("/home/wzh/myProjects/CsiFeedback")
OUT = ROOT / "docs/ideas/metrics/baseline_reeval.json"
CRS = [4, 8, 16, 32, 64]
MODELS = ["clnet", "csinet", "stnet"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_cost2100_loaders(
    data_dir=str(MAIN / "data"), scenario="indoor", batch_size=400,
    num_workers=0, pin_memory=False, load_raw=True,
)


@torch.no_grad()
def run(model):
    preds, gts, raws = [], [], []
    for x, raw in test_loader:
        x = x.to(device)
        preds.append(model(x).cpu())
        gts.append(x.cpu())
        raws.append(raw)
    return torch.cat(preds), torch.cat(gts), torch.cat(raws)


results = {}
for name in MODELS:
    for cr in CRS:
        cfg = load_config(str(ROOT / f"csifeedback/configs/baselines/{name}_indoor_cr{cr}.yaml"))
        cfg.data.data_dir = str(MAIN / "data")
        run_dir = sorted((MAIN / f"experiments/{name}/indoor_cr{cr}").glob("2*"))[-1]
        entry = {}
        for ckpt_name, tag in (("best_nmse", "nmse"), ("best_rho", "rho")):
            model = get_model_from_config(cfg.model).to(device)
            ckpt = load_checkpoint(str(run_dir / f"checkpoints/{ckpt_name}.pth"), map_location=str(device))
            load_model_state_dict(model, ckpt.model_state_dict)
            model.eval()
            pred, gt, raw = run(model)
            entry[f"nmse_{tag}"] = round(compute_nmse(pred, gt).item(), 4)
            entry[f"rho_{tag}"] = round(compute_rho(pred, raw).item(), 6)
            entry["run"] = str(run_dir.relative_to(MAIN))
        results[f"{name}_cr{cr}"] = entry
        print(name, cr, entry, flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
print("written", OUT)
