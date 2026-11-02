# AGENTS.md

本文件是 Codex、Kimi Code、Claude Code 等 coding agent 在本仓库工作时的统一说明。
用户指令优先于本文件；本文件优先于一般默认行为。

## 项目使命

本仓库是 Massive MIMO CSI 反馈的研究平台。当前目标不是复现论文，而是：

> 以 CLNet、STNet、CsiNet 为 baseline，在 COST2100 indoor/outdoor 场景下，
> 研究并实现新的 SOTA CSI 反馈模型。

研究范围固定为：

- 数据集：COST2100，`(N, 2, 32, 32)` 稀疏 CSI
- 压缩比：1/4、1/8、1/16、1/32、1/64
- 指标：NMSE（dB，越低越好）、rho（越高越好）
- 评测：官方 train / val / test 划分

## 非目标与边界

- 不要为了“复现一致性”修改 baseline 的网络结构或训练行为。
- 不要修改 `archive/`，它只是原始项目的历史参考。
- 不要把 `experiments/`、`data/`、`*.pth`、日志提交到 git。
- 不要用测试集调参；模型选择基于验证集，测试集只用于最终报告。
- 不要引入未在 `pyproject.toml` / `requirements.txt` 中声明的重量级依赖，
  除非用户明确同意。

## 环境

- Linux GPU 服务器，Miniforge Conda 环境 `csifeedback`
- Python 3.12，PyTorch 2.11.0+cu128（CUDA 12.8）
- 2 × NVIDIA RTX 5090（32 GB）

```bash
conda activate csifeedback
export CUDA_VISIBLE_DEVICES=0   # 训练前必须指定，否则可能 CUDA 初始化失败
```

运行长任务前先看 GPU 占用：

```bash
nvidia-smi
```

不要终止其他用户的进程。长跑使用 `tmux` / `screen`，不要占用前台终端。

## 分支工作流（必须遵守）

```text
main                        稳定基座，始终测试通过
 ├── idea/<short-name>      一个 idea 一个分支
 └── chore/<topic>          工程维护
```

- 新 idea 一律从最新 `main` 签出 `idea/<short-name>`，不要直接在 `main` 上做研究实验。
- 一个分支只做一个 idea；不要混入无关改动。
- 合并回 `main` 前必须满足 `docs/research_workflow.md` 中的合并标准。
- 工程改动走 `chore/<topic>`，不能改变 baseline 的数值行为。

```bash
git switch main
git pull
git switch -c idea/<short-name>
```

## Baseline 完整性（受保护资产）

以下内容视为 baseline 协议的一部分，改动会破坏与历史结果的对比：

- `csifeedback/models/{clnet,stnet,csinet}.py`
- `csifeedback/trainers/{clnet,stnet,csinet}.py`
- `csifeedback/data/cost2100.py`
- `csifeedback/metrics/csi_metrics.py`
- `csifeedback/configs/baselines/*.yaml`

如果确实必须修改，必须：

1. 在提交说明中写明原因；
2. 重跑全部 30 个 baseline；
3. 更新 `docs/baselines.md` 和 README 的参考表；
4. 注明新旧结果不可直接比较。

新模型必须使用相同的 `data/` 文件、相同的 train/val/test 划分和相同的指标实现。

## 扩展点

### 模型注册

在 `csifeedback/models/<name>.py` 中实现 `CSIAutoencoder` 子类，并用装饰器注册：

```python
from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


@register_model("my_idea")
def create_my_idea(encoded_dim: int = 512, **kwargs) -> CSIAutoencoder:
    return MyIdea(encoded_dim=encoded_dim, **kwargs)
```

然后在 `csifeedback/models/__init__.py` 中导入该模块。不要修改已有模型文件。

### 配置

- baseline：`csifeedback/configs/baselines/`（只读）
- 新 idea：`csifeedback/configs/ideas/<idea>/<idea>_<scenario>_cr<X>.yaml`

`model.extra` 会原样透传给工厂函数，用于 depth、num_heads 等自定义参数。

### Trainer

- 标准 MSE + 单优化器 + Adam/AdamW/SGD + const/cosine：直接用 `DefaultTrainer`，
  不需要注册，也不需要改 `csifeedback/trainers/__init__.py`。
- 自定义 loss、双优化器、特殊更新逻辑：新增专用 trainer，并在
  `csifeedback/trainers/__init__.py` 的 `_TRAINER_MAP` 中注册。
- 最优 checkpoint 由 `training.select_best` 控制：`all`（默认，test rho +
  test NMSE）、`test_rho`、`test_nmse`、`val_loss`。新 idea 推荐 `val_loss`，
  避免用测试集选模型；与 baseline 表严格对比时需说明选择策略。

### 实验摘要

`BaseTrainer` 会在每次运行目录写 `metrics.json`，包含配置、最优 rho/NMSE、
评估历史、耗时、参数量和 git commit。新增训练逻辑时不要绕过它。

## 实验规范

- 快速验证：先跑 `training.epochs=2` 的 smoke test，再跑完整 1000 epoch。
- 完整实验至少覆盖目标场景与目标 CR；关键结论建议 3 个随机种子。
- 新 idea 推荐 `training.select_best: "val_loss"`，不要用测试集选模型；
  如果与 baseline 表对比时使用了不同选择策略，必须在结果中注明。
- 对比使用：

```bash
python scripts/compare.py --latest --min-epochs 1000
```

- 报告结果时写清：随机种子、epoch 数、checkpoint 选择策略、是训练中观察到的最优值
  还是 checkpoint 复评值。
- 负结果也要记录在 `docs/ideas/<name>.md`，不要只留在聊天记录里。

## 测试与完成标准

```bash
pytest tests/ -q
```

新增模型至少要有：

- 模型实例化 / 前向 shape 测试；
- 单 batch 过拟合测试；
- 配置能被 `tests/test_configs.py` 自动加载。

在声称“完成 / 修好 / 通过”之前，必须实际运行相关测试并看到通过输出。
不要只跑了 import 或 compile 就宣称完成。

## Git 规范

- 提交信息使用中文，风格与现有历史一致。
- 除非用户明确要求，不要执行 `git commit`、`git push`、`git reset` 等操作。
- 不要提交 `experiments/`、`data/`、检查点、日志、`__pycache__`。
- 提交前检查 `git status`，确认没有把大文件或运行产物带进来。

## 代码风格

- 保持与现有代码一致：中文 docstring / 注释，类型标注，清晰的模块边界。
- 不要为了新模型修改已有模型文件；通过注册表和 `model.extra` 扩展。
- 不要添加解释“我改了什么”的注释；注释只解释不明显的设计原因。
- 优先复用 `BaseTrainer`、`build_optimizer`、`compute_nmse`、`compute_rho` 等现有设施。

## 文档维护

以下改动需要同步更新文档：

- 新模型 / 新配置：更新 `docs/ideas/<name>.md`，必要时更新 README。
- 训练/评测协议变化：更新 `docs/baselines.md`。
- 工作流变化：更新 `docs/research_workflow.md` 和本文件。
- 目录结构变化：更新 README 的仓库结构。

## 常用命令

```bash
# 测试
pytest tests/ -q

# 训练 baseline
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/baselines/clnet_indoor_cr4.yaml

# 训练所有 baseline
CUDA_VISIBLE_DEVICES=0 python scripts/train_all.py --pattern "baselines/*.yaml"

# 评估
python scripts/evaluate.py \
    --config csifeedback/configs/baselines/clnet_indoor_cr4.yaml \
    --checkpoint experiments/clnet/indoor_cr4/<timestamp>/checkpoints/last.pth \
    --split test

# 汇总对比
python scripts/compare.py --latest --min-epochs 1000
```
