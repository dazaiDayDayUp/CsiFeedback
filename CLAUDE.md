# CLAUDE.md

本文件用于指导 Claude Code（claude.ai/code）在本仓库中开展工作。

## 项目概述

本仓库是三个基于深度学习的 Massive MIMO CSI 反馈模型的统一 PyTorch 复现，目标是长期维护并开源：

- **CLNet** —— 论文 *Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback*（Ji & Li，IEEE WCL 2021）
- **STNet** —— 论文 *A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback*（Mourya & Amuru，arXiv 2022）
- **CsiNet** —— 论文 *Deep Learning for Massive MIMO CSI Feedback*（Wen 等人，IEEE WCL 2018）

原先三个独立的 GitHub 项目已归档到 `archive/`，不再维护。当前 actively maintained 的代码统一在 `csifeedback/` 包中，使用 YAML 配置驱动训练与评估。

当前工作环境为实验室 Linux GPU 服务器，使用 Miniforge 管理的 `csifeedback` Conda 环境（Python 3.12.13），已安装 GPU 版 PyTorch 2.11.0+cu128（CUDA 12.8），显卡为 2 × NVIDIA GeForce RTX 5090（32 GB 显存）。TensorFlow 未安装，原始 Keras/TF 脚本仅作为历史参考保留在 `archive/` 中。

## 环境配置

```bash
# 激活 miniforge 环境
conda activate csifeedback

# 安装依赖（已包含 pytest、pyyaml）
pip install -r requirements.txt
```

当前已安装核心包：PyTorch 2.11.0+cu128、numpy 1.26.4、scipy、matplotlib、tqdm、colorama、thop、pytest、pytest-sugar、pyyaml。

## 目录结构

```
/home/wzh/myProjects/CsiFeedback/
├── csifeedback/          # 统一 Python 包
│   ├── models/           # clnet.py、stnet.py、csinet.py、base.py
│   ├── data/             # cost2100.py（统一数据加载）
│   ├── metrics/          # csi_metrics.py（NMSE + rho）
│   ├── trainers/         # base.py、clnet.py、stnet.py、csinet.py
│   ├── utils/            # config.py、checkpoint.py、device.py、logging.py、scheduler.py
│   └── configs/          # YAML 实验配置
├── scripts/
│   ├── train.py          # 统一训练入口
│   └── evaluate.py       # 统一评估入口
├── tests/                # pytest 单元测试与回归测试
├── experiments/          # 训练输出（按时间戳组织）
├── archive/              # 原始三个项目（只读参考）
├── data/                 # COST2100 .mat 数据
├── docs/
│   ├── migration.md      # 从旧脚本迁移的说明
│   └── architecture.md   # 三个模型的架构说明
├── README.md
├── CLAUDE.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## 数据

COST2100 `.mat` 文件放在 `data/`：

- 稀疏 CSI：`DATA_Htrain{in|out}.mat`、`DATA_Hval{in|out}.mat`、`DATA_Htest{in|out}.mat`，键名 `HT`，形状 `(N, 2, 32, 32)`。
- 原始 CSI：`DATA_HtestF{in|out}_all.mat`，键名 `HF_all`，形状 `(N, 32, 125, 2)`。

配置文件默认使用 `./data`（相对于仓库根目录）。

## 运行方式

### 训练

```bash
python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml
```

根据服务器负载可覆盖 batch size：

```bash
python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml \
    -o data.batch_size=64
```

覆盖任意配置项：

```bash
python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml \
    -o training.epochs=50 data.batch_size=16
```

训练时会显示：
- 每个 epoch 的 `tqdm` 进度条（带颜色）
- 当前 batch loss、学习率、速度、ETA
- epoch 结束后的本轮/累计/预计剩余用时

### 评估

```bash
python scripts/evaluate.py \
    --config csifeedback/configs/clnet_indoor_cr4.yaml \
    --checkpoint experiments/clnet/indoor_cr4/YYYYMMDD_HHMMSS/checkpoints/last.pth \
    --split test
```

### 测试

```bash
pytest tests/
```

已安装 `pytest-sugar`，测试运行时会显示彩色进度条。`tests/conftest.py` 会在全部测试结束后打印模块总结，包含 ✅/❌ 状态和每个测试文件的中文说明。当前测试状态：**67 passed，0 skipped**。

## 关键模块

- `csifeedback/models/clnet.py` —— CLNet 自编码器，保留 `hsigmoid`、双重 xavier 初始化。
- `csifeedback/models/stnet.py` —— STNet，保留 decoder 中 `out` 被两个 WTL 循环复用的 broadcast 行为。
- `csifeedback/models/csinet.py` —— CsiNet，encoder conv 使用 `bias=True`，residual conv 使用 `bias=False`。
- `csifeedback/data/cost2100.py` —— 统一数据加载，同时支持 `indoor`/`outdoor` 和 `in`/`out` 两种场景写法。
- `csifeedback/metrics/csi_metrics.py` —— 统一 NMSE 与 rho，rho 通过 FFT 零填充到 257 bins 再切片到 125 bins。
- `csifeedback/trainers/` —— `BaseTrainer` 提供通用循环；三个子类分别处理 CLNet 单优化器+cosine、STNet 双优化器、CsiNet val-based best save。
- `csifeedback/utils/config.py` —— dataclass + YAML + 命令行覆盖。

## 压缩比配置

| 模型   | CR=1/4             | 1/8            | 1/16           | 1/32          | 1/64          |
|--------|--------------------|----------------|----------------|---------------|---------------|
| CLNet  | `reduction: 4`     | `reduction: 8` | `reduction: 16`| `reduction: 32`| `reduction: 64`|
| STNet  | `encoded_dim: 512` | `encoded_dim: 256` | `encoded_dim: 128` | `encoded_dim: 64` | `encoded_dim: 32` |
| CsiNet | `encoded_dim: 512` | `encoded_dim: 256` | `encoded_dim: 128` | `encoded_dim: 64` | `encoded_dim: 32` |

## 训练检查点

训练产生的检查点按时间戳保存在 `experiments/{model_name}/{scenario}_cr{X}/YYYYMMDD_HHMMSS/checkpoints/` 下：

- `last.pth`：训练结束时的最终模型
- `best_rho.pth`：测试集 rho 最优的模型
- `best_nmse.pth`：测试集 NMSE 最优的模型
- `epoch{N}.pth`：每隔 `checkpoint_freq` 个 epoch 保存的阶段性模型

评估时使用 `scripts/evaluate.py` 加载上述任意检查点。

## 重要注意事项

- **原始 `CLNet/`、`STNet/`、`CsiNet/` 目录已删除**，完整历史版本保留在 `archive/`，归档时仅保留源码与 README，中间产物已清理。
- **TensorFlow 未安装**，不要尝试运行 `archive/CsiNet/` 下的 Keras 脚本。
- **GPU 服务器环境**：当前使用实验室 Linux 服务器与 Miniforge `csifeedback` 环境；批量训练建议使用 `scripts/train_all.py`。
- **模型层定义已原样保留**，重构只改变工程组织，不改变网络结构。
- **超参数已与原始仓库对齐**：CLNet cosine scheduler 使用 `lr=0.002`，warmup 从 0 开始；STNet 使用双 Adam `betas=(0.5, 0.999)`；CsiNet 使用单 Adam 并按 val loss 保存最优。
- **开闭原则**：添加新模型只需在 `csifeedback/models/__init__.py` 和 `csifeedback/trainers/__init__.py` 注册，通过 `model.extra` 传递自定义参数，无需修改已有模型代码。

## 论文参考指标

- CLNet indoor：1/4 约 -29.16 dB，1/64 约 -6.34 dB。
- STNet indoor：1/4 约 -31.81 dB，1/64 约 -7.81 dB。
- CsiNet indoor：1/4 约 -17.36 dB，1/16 约 -8.65 dB，1/32 约 -6.24 dB，1/64 约 -5.84 dB。

目前仅做了短周期验证，指标距离论文值还很远，原因是训练周期过短，与模型实现无关。
