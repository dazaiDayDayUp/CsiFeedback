# CSI Feedback

三个基于深度学习的 Massive MIMO CSI 反馈模型的统一 PyTorch 复现：

- **CLNet** —— 论文 *Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback*（Ji & Li，IEEE WCL 2021）
- **STNet** —— 论文 *A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback*（Mourya & Amuru，arXiv 2022）
- **CsiNet** —— 论文 *Deep Learning for Massive MIMO CSI Feedback*（Wen 等人，IEEE WCL 2018）

本仓库将原先三个独立下载的 GitHub 项目整理为单一、配置驱动、符合开闭原则的统一包。

## 快速开始

```bash
# 激活本地虚拟环境
source venv/Scripts/activate   # Git Bash / POSIX shell on Windows

# 训练 CLNet indoor CR=1/4（本地 2GB GPU 需要减小 batch size）
python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml \
    -o data.batch_size=16

# 在服务器等大显存机器上训练（可使用默认 batch_size=200）
python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml

# 训练时覆盖任意配置项
python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml \
    -o training.epochs=50 data.batch_size=16

# 评估已保存的检查点
python scripts/evaluate.py \
    --config csifeedback/configs/clnet_indoor_cr4.yaml \
    --checkpoint experiments/clnet/indoor_cr4/YYYYMMDD_HHMMSS/checkpoints/last.pth
```

## 仓库结构

```
.
├── csifeedback/          # 统一 Python 包
│   ├── models/           # clnet、stnet、csinet
│   ├── data/             # 统一 COST2100 数据加载器
│   ├── metrics/          # 统一 NMSE + rho 计算
│   ├── trainers/         # 统一训练循环
│   ├── utils/            # 配置、检查点、设备、日志、学习率调度
│   └── configs/          # YAML 实验配置
├── scripts/              # train.py、evaluate.py
├── tests/                # 单元测试与回归测试
├── experiments/          # 训练输出（按时间戳组织）
├── archive/              # 原始项目代码（只读参考）
├── data/                 # COST2100 .mat 数据文件
├── docs/
│   ├── migration.md      # 从旧脚本迁移的说明
│   └── architecture.md   # 三个模型的架构说明
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## 支持的压缩比

| 模型   | CR=1/4 | 1/8 | 1/16 | 1/32 | 1/64 |
|--------|--------|-----|------|------|------|
| CLNet  | `reduction: 4`  | 8   | 16   | 32   | 64   |
| STNet  | `encoded_dim: 512` | 256 | 128  | 64   | 32   |
| CsiNet | `encoded_dim: 512` | 256 | 128  | 64   | 32   |

压缩比与模型参数的详细对应关系见 [`docs/architecture.md`](docs/architecture.md)。

## 当前状态

- **配置矩阵**：已补全 30 个 YAML 配置（3 模型 × 2 场景 × 全部压缩比，CsiNet 额外包含 1/8）。
- **测试**：`pytest tests/` 全部通过（67 passed，0 skipped），覆盖模型、数据、指标、配置加载、单 batch 过拟合、OCP 扩展点。
- **实现一致性**：已完成与三个原始仓库的全面对照，修复了 CLNet cosine warmup 学习率起点不一致的问题；网络结构、数据预处理、指标计算与原始实现一致。
- **训练验证**：已在 GPU 上跑通 CLNet indoor CR=1/4 的 1 epoch smoke test（batch_size=16，NMSE 4.87 dB，rho 0.419），距离论文值还很远，原因是训练周期过短。
- **完整训练**：待迁移到更大显存的 GPU 服务器后批量执行。

## 主要特性

- **配置驱动**：所有实验通过 YAML 配置管理，命令行可覆盖任意字段。
- **统一入口**：`scripts/train.py` 和 `scripts/evaluate.py` 支持三个模型。
- **训练可视化**：每个 epoch 带 `tqdm` 进度条，显示当前 loss、学习率、速度、ETA；epoch 结束后显示本轮/累计/预计剩余用时。
- **开闭原则**：新模型只需在 `models/` 和 `trainers/` 注册工厂，通过 `model.extra` 传递自定义参数，无需修改已有模型代码。
- **检查点管理**：自动保存 `last.pth`、`best_rho.pth`、`best_nmse.pth`（CLNet/STNet）或 `best_val.pth`（CsiNet），以及按周期的 `epoch{N}.pth`。

## 数据

COST2100 数据集应放置在仓库根目录的 `data/` 下：

```
data/
├── DATA_Htrainin.mat
├── DATA_Hvalin.mat
├── DATA_Htestin.mat
├── DATA_HtestFin_all.mat
├── DATA_Htrainout.mat
├── DATA_Hvalout.mat
├── DATA_Htestout.mat
└── DATA_HtestFout_all.mat
```

## 测试

```bash
pytest tests/
```

测试覆盖模型实例化、前向 shape、确定性前向、指标正确性、单 batch 过拟合等，全部可在 CPU 上运行，无需完整训练。

## 旧代码归档

原始三个独立项目已完整归档到 `archive/`，仅作历史参考，不再维护。本项目使用统一入口 `scripts/train.py` 和 `scripts/evaluate.py` 从头训练与评估。

训练产生的检查点按时间戳保存在 `experiments/{model_name}/{scenario}_cr{X}/YYYYMMDD_HHMMSS/checkpoints/` 下。

## 硬件说明

当前开发环境：

- Windows 11 笔记本
- Python 3.12
- PyTorch 2.2.2（CUDA 11.8）
- NVIDIA GeForce MX450（2GB 显存）

在 2GB 显存上训练时，需要将 `batch_size` 降到 16–32。完整 1000 epoch 训练建议在更大显存的 GPU 工作站或服务器上运行。CPU-only 环境也可以运行 smoke test，但完整训练非常慢。

## 引用

如果在研究中使用本代码，请引用原始论文：

```bibtex
@article{ji2021clnet,
  title={Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback},
  author={Ji, Xiangjun and Li, Chao-Kai},
  journal={IEEE Wireless Communications Letters},
  year={2021}
}

@article{wen2018csinet,
  title={Deep Learning for Massive MIMO CSI Feedback},
  author={Wen, Chao-Kai and Shih, Wen-Tai and Jin, Shi},
  journal={IEEE Wireless Communications Letters},
  year={2018}
}
```
