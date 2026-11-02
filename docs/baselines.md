# Baseline 协议与参考结果

本文档固定 CLNet、STNet、CsiNet 三个 baseline 的训练/评测协议，作为新 idea
公平对比的参照。baseline 代码、数据划分与指标实现属于**受保护资产**：
除非有明确理由并重跑全部 baseline，否则不要修改。

## Baseline 代码位置

| 模型 | 网络 | 训练器 | 配置 |
|------|------|--------|------|
| CLNet | `csifeedback/models/clnet.py` | `csifeedback/trainers/clnet.py` | `csifeedback/configs/baselines/clnet_*.yaml` |
| STNet | `csifeedback/models/stnet.py` | `csifeedback/trainers/stnet.py` | `csifeedback/configs/baselines/stnet_*.yaml` |
| CsiNet | `csifeedback/models/csinet.py` | `csifeedback/trainers/csinet.py` | `csifeedback/configs/baselines/csinet_*.yaml` |

三个模型都继承 `CSIAutoencoder`，输入输出均为 `(N, 2, 32, 32)`。

## 训练协议

| 项目 | 设定 |
|------|------|
| 数据集 | COST2100，indoor / outdoor |
| 训练样本 | `DATA_Htrain{in,out}.mat` |
| 验证样本 | `DATA_Hval{in,out}.mat` |
| 测试样本 | `DATA_Htest{in,out}.mat` + `DATA_HtestF{in,out}_all.mat`（算 rho） |
| batch size | 200 |
| epochs | 1000 |
| 损失 | MSE |
| 随机种子 | `null`（不做固定种子；复现实验时建议显式设置 `training.seed`） |
| 评估频率 | `val_freq=10`、`test_freq=10` |
| 检查点频率 | 每 50 epoch |
| 最优模型 | `training.select_best: "all"`：test rho 最优 + test NMSE 最优（CsiNet 额外按 val loss） |

模型特定超参数：

| 模型 | 优化器 | 学习率 | 调度 |
|------|--------|--------|------|
| CLNet | Adam，betas=(0.9, 0.999) | 2e-3 | cosine warmup（30 epoch warmup，退火到 5e-5） |
| STNet | 双 Adam（encoder/decoder 分离），betas=(0.5, 0.999) | 2e-3 | const |
| CsiNet | Adam，betas=(0.9, 0.999) | 1e-3 | const |

压缩比映射（输入维度 2048）：

| CR | CLNet `reduction` | STNet / CsiNet `encoded_dim` |
|----|-------------------|------------------------------|
| 1/4 | 4 | 512 |
| 1/8 | 8 | 256 |
| 1/16 | 16 | 128 |
| 1/32 | 32 | 64 |
| 1/64 | 64 | 32 |

## 本地参考结果

本机已完成的 30 个 1000 epoch 运行，测试集上观察到的最优值。
NMSE 单位为 dB（越低越好），rho 为余弦相似度（越高越好）。

### Indoor

| CR | CLNet NMSE / rho | STNet NMSE / rho | CsiNet NMSE / rho |
|----|------------------|------------------|-------------------|
| 1/4 | -27.87 / 0.9961 | **-30.17 / 0.9967** | -14.45 / 0.9807 |
| 1/8 | -15.57 / 0.9836 | **-19.28 / 0.9914** | -11.12 / 0.9600 |
| 1/16 | -10.68 / 0.9559 | **-14.75 / 0.9812** | -8.73 / 0.9348 |
| 1/32 | -8.95 / 0.9345 | **-9.51 / 0.9433** | -8.50 / 0.9281 |
| 1/64 | -6.04 / 0.8699 | **-6.82 / 0.8917** | -4.34 / 0.7991 |

### Outdoor

| CR | CLNet NMSE / rho | STNet NMSE / rho | CsiNet NMSE / rho |
|----|------------------|------------------|-------------------|
| 1/4 | -5.90 / 0.8671 | **-12.62 / 0.9516** | -10.38 / 0.9338 |
| 1/8 | -8.00 / 0.8989 | **-8.17 / 0.9007** | -6.97 / 0.8684 |
| 1/16 | **-5.36 / 0.8259** | -5.44 / 0.8270 | -4.89 / 0.8079 |
| 1/32 | -3.43 / 0.7298 | **-3.52 / 0.7290** | -3.05 / 0.6841 |
| 1/64 | **-2.25 / 0.6328** | -1.91 / 0.5862 | -1.36 / 0.5541 |

> 说明：以上数值来自训练过程中 `test_freq=10` 记录的最优值，不是重新加载
> `best_*.pth` 后的复评。若要用于论文对比，建议用 `scripts/evaluate.py`
> 对固定 checkpoint 重新评测，并固定随机种子、重复多次取均值。

> **选择策略**：baseline 默认 `select_best: "all"`，即按测试集指标保存最优
> checkpoint。新 idea 推荐使用 `select_best: "val_loss"`，只用验证集选模型。
> 如果两边策略不同，比较时需要在结果中注明；严格对比时建议用相同策略重跑
> baseline。

已知现象：

- CLNet outdoor 1/4 的结果明显差于 1/8。这更可能是该配置下的训练稳定性/过拟合
  问题，而不是实现错误；新模型对比时不要把这个异常点当作强 baseline。
- 本地 CLNet / CsiNet 的 indoor NMSE 低于部分论文报告值，主要来自训练协议、
  数据划分细节和评测方式差异。跨论文比较时以协议一致为准。

### 论文参考值（indoor，NMSE dB）

| CR | CLNet | STNet | CsiNet |
|----|-------|-------|--------|
| 1/4 | -29.16 | -31.81 | -17.36 |
| 1/8 | — | — | — |
| 1/16 | — | — | -8.65 |
| 1/32 | — | — | -6.24 |
| 1/64 | -6.34 | -7.81 | -5.84 |

> 论文值仅供方向性参考。不同论文的训练细节、数据预处理和 rho/NMSE 实现可能不同。

## 复现命令

```bash
conda activate csifeedback
export CUDA_VISIBLE_DEVICES=0

# 全部 baseline（30 个配置）
python scripts/train_all.py --pattern "baselines/*.yaml"

# 单个 baseline
python scripts/train.py \
    --config csifeedback/configs/baselines/stnet_indoor_cr4.yaml

# 固定种子重跑（推荐用于论文表格）
python scripts/train.py \
    --config csifeedback/configs/baselines/stnet_indoor_cr4.yaml \
    -o training.seed=0
```

## 刷新参考结果

```bash
# 只保留每个 (model, scenario, CR) 的最新完整运行
python scripts/compare.py --latest --min-epochs 1000 --format markdown
```

更新本文档时，请同时更新 README 中的简表，并记录：

- 运行时间戳目录
- git commit
- 是否固定种子
- 是训练中观察到的最优值，还是 checkpoint 复评值

## Baseline 完整性规则

1. **不要修改** baseline 的网络结构、初始化、优化器超参数和训练循环，
   除非目标是修复明确的实现错误。
2. **不要修改** 数据加载、划分方式、归一化方式和指标实现。任何改动都会破坏
   与历史结果的对比。
3. 如果确实修改了上述内容，必须：
   - 在 PR/提交说明中写清原因；
   - 重跑全部 30 个 baseline；
   - 更新 `docs/baselines.md` 与 README 的参考表；
   - 在文档中注明新旧结果不可直接比较。
4. 新 idea 必须使用相同的 `data/` 文件、相同的 train/val/test 划分和相同的
   `compute_nmse` / `compute_rho`。
5. 不要用测试集调参。模型选择应基于验证集；测试集只用于最终报告。
