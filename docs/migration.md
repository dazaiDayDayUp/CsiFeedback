# 迁移指南

本文档帮助原独立脚本的用户迁移到统一的 `csifeedback` 包。

> 快速参考：压缩比速查表见 [`README.md`](../README.md)，模型参数与架构细节见 [`architecture.md`](architecture.md)。

## 前后对比

| 任务 | 旧方式 | 新方式 |
|------|--------|--------|
| 训练 CLNet | `cd CLNet && python main.py --data-dir ../data ...` | `python scripts/train.py --config csifeedback/configs/clnet_indoor_cr4.yaml` |
| 训练 STNet | `cd STNet && python stnet_fixed.py` | `python scripts/train.py --config csifeedback/configs/stnet_indoor_cr4.yaml` |
| 训练 CsiNet | `cd CsiNet && python csinet_pytorch.py` | `python scripts/train.py --config csifeedback/configs/csinet_indoor_cr4.yaml` |
| 评估 | 各模型专用脚本 | `python scripts/evaluate.py --config ... --checkpoint ...` |

## 配置覆盖

任何 YAML 字段都可以从命令行覆盖：

```bash
python scripts/train.py \
    --config csifeedback/configs/clnet_indoor_cr4.yaml \
    -o training.epochs=50 data.batch_size=16
```

> 提示：可根据服务器显存与负载调整 `data.batch_size` 等训练参数。

## 训练检查点

训练产生的检查点按时间戳保存在 `experiments/{model_name}/{scenario}_cr{X}/YYYYMMDD_HHMMSS/checkpoints/` 下：

- `last.pth`：训练结束时的最终模型
- `best_rho.pth`：测试集 rho 最优的模型
- `best_nmse.pth`：测试集 NMSE 最优的模型
- `epoch{N}.pth`：每隔 `checkpoint_freq` 个 epoch 保存的阶段性模型

评估时使用 `scripts/evaluate.py` 加载上述任意检查点：

```bash
python scripts/evaluate.py \
    --config csifeedback/configs/clnet_indoor_cr4.yaml \
    --checkpoint experiments/clnet/indoor_cr4/YYYYMMDD_HHMMSS/checkpoints/last.pth \
    --split test
```

## 底层变化

1. **数据加载**统一在 `csifeedback/data/cost2100.py`。三个模型都通过同一段代码读取相同的 `.mat` 文件。
2. **评估指标**统一在 `csifeedback/metrics/csi_metrics.py`。NMSE 对三个模型一致计算；rho（FFT 扩展到 125 bins）现在也对 CsiNet 和 STNet 可用。
3. **训练循环**统一在 `csifeedback/trainers/`。模型差异（如 STNet 的双优化器）隔离在子类中。
4. **网络定义**迁移时改动最小。层定义、激活函数、初始化均原样保留。
5. **原始脚本**已归档到 `archive/`，仅保留源码与 README，中间产物已清理。

## 必须保留的原始行为

- STNet decoder 复用同一个 `out` 张量参与两个 WTL 循环（包含 shape broadcast），必须原样保留。
- CLNet 使用 `hsigmoid`（即 `F.relu6(x+3)/6`），不能替换为普通 Sigmoid。
- CsiNet encoder conv 使用 `bias=True`，residual block conv 使用 `bias=False`，必须保留。
- NMSE 计算需从实部/虚部各减去 0.5（数据零点）。
- rho 计算需将 32 bins 零填充到 257 bins，再切片到 125 bins，与 CLNet 原实现一致。

## 添加新模型

本包遵循开闭原则。添加新模型时，**不需要修改任何已有模型代码**，只需在扩展点注册：

1. 在 `csifeedback/models/<new_model>.py` 中实现新网络，继承 `CSIAutoencoder`。
2. 在 `csifeedback/models/__init__.py` 中注册工厂函数。
3. 在 `csifeedback/trainers/<new_model>.py` 中实现训练器（或复用 `BaseTrainer`）。
4. 在 `csifeedback/trainers/__init__.py` 中注册训练器。
5. 在 `csifeedback/configs/` 下添加 YAML 配置。

### 自定义模型参数

`ModelConfig` 已通过 `model.extra` 字段支持任意附加参数。例如新模型需要 `depth` 和 `num_heads`：

```yaml
model:
  name: "mynet"
  encoded_dim: 256
  extra:
    depth: 2
    num_heads: 8
```

工厂函数会收到 `encoded_dim=256, depth=2, num_heads=8`。原有 CLNet / STNet / CsiNet 不需要 `extra`。

无需修改任何已有模型代码。
