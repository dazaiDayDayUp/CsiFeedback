# CSI Feedback Research Platform

面向 Massive MIMO CSI 反馈的研究平台，目标是产出新 SOTA 模型与论文级实验证据，
支撑论文发表。

本仓库以 CLNet、STNet、CsiNet 为 baseline，在 COST2100 indoor/outdoor 场景下，
用统一的数据划分、指标实现、训练与评测流程开展公平对比。研究 idea 在独立分支上
实现和验证，最终形成可复现的 SOTA 结果、消融实验与论文材料。

## 研究范围

| 项目 | 设定 |
|------|------|
| 任务 | Massive MIMO 下行 CSI 反馈（自编码器压缩与重建） |
| 数据集 | COST2100（indoor / outdoor） |
| 输入 | 稀疏 CSI，`(N, 2, 32, 32)`（实部/虚部，已归一化到 `[0, 1]`） |
| 压缩比 | 1/4、1/8、1/16、1/32、1/64 |
| 指标 | NMSE（dB，越低越好）、rho（余弦相似度，越高越好） |
| 评测划分 | COST2100 官方 train / val / test |

所有模型共享同一套数据加载器、指标实现和训练循环，保证 baseline 与新 idea 的对比公平。

## 项目定位

- **`main` 是稳定基座**：始终保持可训练、可评测、测试通过。
- **每个 idea 一个 feature 分支**：从 `main` 签出 `idea/<name>`，互不干扰。
- **baseline 是公平对比的固定参照**：CLNet / STNet / CsiNet 的网络结构、训练行为和
  评测协议保持稳定；如果确实必须改动，需要重跑全部 baseline 并更新基线表。
- **idea 配置隔离**：新实验放在 `csifeedback/configs/ideas/<idea>/`，
  与 `baselines/` 分开。
- **结果结构化**：每次运行自动写 `metrics.json`，可直接与 baseline 对比。

## 研究产出

每个 idea 的目标是形成论文级证据，而不只是跑通代码：

- 与 baseline 在相同场景、相同 CR 矩阵下的公平 SOTA 对比
- 至少 3 个随机种子的均值 ± 标准差
- 关键模块与训练策略的消融实验
- 参数量、FLOPs / 推理开销
- 可复现的配置、提交、`metrics.json` 与检查点

论文级实验清单见 [`docs/paper_checklist.md`](docs/paper_checklist.md)。

## 快速开始

```bash
# 激活环境
conda activate csifeedback

# 安装依赖（首次）
pip install -r requirements.txt

# 训练一个 baseline（必须通过 CUDA_VISIBLE_DEVICES 指定 GPU）
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/baselines/clnet_indoor_cr4.yaml

# 评估某个检查点
python scripts/evaluate.py \
    --config csifeedback/configs/baselines/clnet_indoor_cr4.yaml \
    --checkpoint experiments/clnet/indoor_cr4/YYYYMMDD_HHMMSS/checkpoints/last.pth \
    --split test

# 汇总所有实验，只保留每个 (model, scenario, CR) 的完整运行
python scripts/compare.py --latest --min-epochs 100
```

> **GPU 注意**：训练前必须用 `CUDA_VISIBLE_DEVICES=0`（或 `1`）指定显卡，
> 否则可能因 CUDA 初始化失败而报错。

## Baseline 参考结果

以下是本机已完成的 1000 epoch 训练中，测试集上观察到的最优 NMSE（dB）。
这不是重新加载 checkpoint 的复评结果，而是训练期间 `test_freq=10` 记录的最优值。

**Indoor**

| CR | CLNet | STNet | CsiNet |
|----|-------|-------|--------|
| 1/4 | -27.87 | **-30.17** | -14.45 |
| 1/8 | -15.57 | **-19.28** | -11.12 |
| 1/16 | -10.68 | **-14.75** | -8.73 |
| 1/32 | -8.95 | **-9.51** | -8.50 |
| 1/64 | -6.04 | **-6.82** | -4.34 |

**Outdoor**

| CR | CLNet | STNet | CsiNet |
|----|-------|-------|--------|
| 1/4 | -5.90 | **-12.62** | -10.38 |
| 1/8 | -8.00 | **-8.17** | -6.97 |
| 1/16 | **-5.36** | -5.44 | -4.89 |
| 1/32 | -3.43 | **-3.52** | -3.05 |
| 1/64 | **-2.25** | -1.91 | -1.36 |

完整协议、论文参考值和已知偏差见 [`docs/baselines.md`](docs/baselines.md)。

## 研究模型：AxisCodec（`idea/cylinder_codec`）

物理出发点：COST2100 的角时延表示是一个「半开圆柱」——角度轴是圆周（均匀线阵的
阵列流形以 1 为周期，bin 31 与 bin 0 相邻），时延轴是半直线（时延非负、能量沿 tap
单调衰减，无平移不变性）。AxisCodec 把这两条几何写成**零参数硬约束**：角度轴用
DFT 域循环卷积（逐通道 32 参数），时延轴用稠密线性混合（每个 tap 独立读出头）；
解码器为「线性基座 + 非线性残差」；训练目标 `MetricMSELoss` 与报告指标
（逐用户归一化 NMSE）只差一个 log10，且 checkpoint 选择与训练目标解耦。

复杂度与 STNet 同量级：CR1/4 时 **2.13M 参数 / 0.022 GFLOPs/样本**
（STNet 2.13M 有效参数 / 0.014 GFLOPs）。

**Indoor CR1/4（`select_best: val_loss`，完全不使用测试集选模型）**

| 模型 | epoch | NMSE (dB) | rho | checkpoint 选择 |
|---|---|---|---|---|
| STNet（固定 checkpoint 复评） | 1000 | -30.174 | 0.99669 | test（`all`） |
| AxisCodec | 300 | -28.984 | 0.99639 | val |
| AxisCodec | 1000 | 见下节/`docs/ideas/cylinder_codec.md` | | val |

完整的动机、负结果（路径参数化、谐波子空间、团块、PCA 初始化）与误差分解见
[`docs/ideas/cylinder_codec.md`](docs/ideas/cylinder_codec.md)。

## 分支与实验工作流

```text
main                        稳定基座：baseline + 统一实验设施
 └── idea/<short-name>      一个 idea 一个分支，独立实验、独立配置
      └── 结果达标后再合并回 main
```

```bash
# 1. 从稳定基座签出 idea 分支
git switch main
git pull
git switch -c idea/dual_domain_transformer

# 2. 实现模型 + 配置（见下文）
# 3. 快速验证
pytest tests/ -q
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/dual_domain_transformer/dual_domain_transformer_indoor_cr4.yaml \
    -o training.epochs=2

# 4. 完整训练 + 与 baseline 对比
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/dual_domain_transformer/dual_domain_transformer_indoor_cr4.yaml
python scripts/compare.py --latest --min-epochs 100
```

详细的 idea 生命周期、命名规范、公平对比与合并标准见
[`docs/research_workflow.md`](docs/research_workflow.md)。

## 仓库结构

```text
.
├── csifeedback/
│   ├── models/
│   │   ├── base.py              # CSIAutoencoder 接口
│   │   ├── registry.py          # @register_model 模型注册表
│   │   ├── clnet.py             # baseline
│   │   ├── stnet.py             # baseline
│   │   └── csinet.py            # baseline
│   ├── trainers/
│   │   ├── base.py              # 通用训练循环 + metrics.json
│   │   ├── default.py           # 配置驱动的新模型通用 trainer
│   │   ├── clnet.py             # baseline 专用训练行为
│   │   ├── stnet.py
│   │   └── csinet.py
│   ├── data/cost2100.py         # 统一数据加载
│   ├── metrics/csi_metrics.py   # 统一 NMSE / rho
│   ├── utils/                   # 配置、检查点、设备、日志、调度、结果
│   └── configs/
│       ├── baselines/           # 30 个 baseline YAML（只读参考）
│       └── ideas/               # 每个 idea 一个子目录
├── scripts/
│   ├── train.py                 # 统一训练入口
│   ├── evaluate.py              # 统一评估入口（可输出 JSON）
│   ├── train_all.py             # 递归批量训练
│   └── compare.py               # 汇总/对比实验
├── tests/                       # pytest 测试
├── experiments/                 # 运行产物（gitignored）
├── data/                        # COST2100 .mat（gitignored）
├── docs/
│   ├── research_workflow.md     # idea 分支与实验规范
│   ├── baselines.md             # baseline 协议与参考结果
│   ├── architecture.md          # baseline 网络结构
│   ├── paper_checklist.md       # 论文级实验与写作清单
│   ├── ideas/TEMPLATE.md        # idea 记录模板
│   └── survey/                  # CSI 反馈文献调研（分类、时间线、可比性）
├── AGENTS.md                    # 所有 coding agent 的统一协作说明
└── README.md
```

## 添加新模型 / 新 idea

### 1. 实现模型并注册

在 `csifeedback/models/` 下新建模块，继承 `CSIAutoencoder`，并用
`@register_model` 注册工厂：

```python
# csifeedback/models/dual_domain_transformer.py
from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


class DualDomainTransformer(CSIAutoencoder):
    def __init__(self, encoded_dim: int = 512, depth: int = 2):
        super().__init__()
        self._encoded_dim = encoded_dim
        self.depth = depth

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> dict:
        return self.encoder.state_dict()

    def get_decoder_state(self) -> dict:
        return self.decoder.state_dict()

    def forward(self, x):
        ...


@register_model("dual_domain_transformer")
def create_dual_domain_transformer(encoded_dim: int = 512, **kwargs):
    return DualDomainTransformer(encoded_dim=encoded_dim, **kwargs)
```

然后在 `csifeedback/models/__init__.py` 中导入该模块，使注册生效：

```python
from csifeedback.models import dual_domain_transformer as _dual_domain_transformer  # noqa: F401
```

已有模型代码不需要改动。`model.extra` 中的任意字段会透传给工厂函数：

```yaml
model:
  name: "dual_domain_transformer"
  encoded_dim: 512
  extra:
    depth: 3
```

### 2. 添加配置

```yaml
# csifeedback/configs/ideas/dual_domain_transformer/dual_domain_transformer_indoor_cr4.yaml
model:
  name: "dual_domain_transformer"
  encoded_dim: 512
  extra:
    depth: 3
data:
  data_dir: "./data"
  scenario: "indoor"
  batch_size: 200
optimizer:
  name: "adam"
  lr: 2.0e-3
scheduler:
  name: "cosine"
training:
  epochs: 1000
  checkpoint_dir: "./experiments"
  select_best: "val_loss"
```

### 3. 训练器

如果新模型使用标准 MSE + 单优化器 + Adam/AdamW/SGD，**不需要写 trainer**：
未注册的模型会自动回退到 `DefaultTrainer`，由配置驱动优化器和调度器。

只有需要自定义 loss、双优化器、特殊参数更新时，才实现专用 trainer，并在
`csifeedback/trainers/__init__.py` 的 `_TRAINER_MAP` 中注册。

`training.select_best` 控制最优 checkpoint 的选择依据：

| 取值 | 行为 |
|------|------|
| `all`（默认） | 同时保存 test rho 最优和 test NMSE 最优，与 CLNet/STNet 一致 |
| `test_rho` | 只保存 test rho 最优 |
| `test_nmse` | 只保存 test NMSE 最优 |
| `val_loss` | 只保存验证损失最优（推荐新 idea 使用，避免测试集选模型） |

无论选择哪种策略，`metrics.json` 都会记录完整的验证/测试评估记录。

### 4. 测试与快速验证

```bash
pytest tests/ -q

# 2 个 epoch 的 smoke test
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/<idea>/<idea>_indoor_cr4.yaml \
    -o training.epochs=2 training.checkpoint_freq=0
```

## 实验产物与对比

每次训练自动创建：

```text
experiments/{model}/{scenario}_cr{X}/{YYYYMMDD_HHMMSS}/
├── checkpoints/
│   ├── last.pth
│   ├── best_rho.pth
│   ├── best_nmse.pth
│   ├── best_val.pth        # CsiNet 或 select_best=val_loss
│   └── epoch{N}.pth
├── logs/train.log
└── metrics.json            # 结构化运行摘要
```

`metrics.json` 包含配置、最优 rho/NMSE、`select_best` 策略、实际被选中的
checkpoint 指标（`selected`）、评估记录、耗时、参数量和 git commit。
`scripts/compare.py` 会优先读取它；对早期没有 `metrics.json` 的运行，会回退解析
`train.log`，此时 `select` 列显示 `unknown`。

```bash
# 表格：按 NMSE 从优到劣
python scripts/compare.py --latest --min-epochs 100

# 只看 indoor、指定模型
python scripts/compare.py --models clnet stnet csinet --scenarios indoor

# Markdown / CSV
python scripts/compare.py --format markdown
python scripts/compare.py --format csv > results.csv
```

## 测试

```bash
pytest tests/
```

测试覆盖模型实例化、前向 shape、指标正确性、数据加载、配置矩阵、
单 batch 过拟合、模型注册、通用 trainer 与 `metrics.json` 落盘。
所有测试均可在 CPU 上运行。

## 环境

- Linux GPU 服务器（实验室）
- Miniforge，Conda 环境 `csifeedback`
- Python 3.12
- PyTorch 2.11.0+cu128（CUDA 12.8）
- 2 × NVIDIA GeForce RTX 5090（32 GB）

训练前：

```bash
conda activate csifeedback
export CUDA_VISIBLE_DEVICES=0
```

## 文档索引

- [`docs/research_workflow.md`](docs/research_workflow.md) —— idea 分支、实验与合并规范
- [`docs/baselines.md`](docs/baselines.md) —— baseline 协议、参考结果与完整性规则
- [`docs/architecture.md`](docs/architecture.md) —— 三个 baseline 的网络结构
- [`docs/paper_checklist.md`](docs/paper_checklist.md) —— 论文级实验与写作清单
- [`docs/survey/README.md`](docs/survey/README.md) —— CSI 反馈文献调研索引与总览
- [`AGENTS.md`](AGENTS.md) —— coding agent 协作说明

## 引用

如果研究中使用本代码，请引用三个 baseline 的原始论文：

```bibtex
@article{ji2021clnet,
  title={Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback},
  author={Ji, Xiangjun and Li, Chao-Kai},
  journal={IEEE Wireless Communications Letters},
  year={2021}
}

@article{mourya2022stnet,
  title={A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback},
  author={Mourya, Rahul and Amuru, SaiDhiraj},
  journal={arXiv preprint arXiv:2201.04605},
  year={2022}
}

@article{wen2018csinet,
  title={Deep Learning for Massive MIMO CSI Feedback},
  author={Wen, Chao-Kai and Shih, Wen-Tai and Jin, Shi},
  journal={IEEE Wireless Communications Letters},
  year={2018}
}
```
