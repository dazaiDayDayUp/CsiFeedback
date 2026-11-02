# 研究 idea 工作流

本文档约定从 baseline 基座出发，研究、实验、合并新模型的标准流程。
目标是让每个 idea 相互隔离、可复现、可比较，并且随时能回到稳定的 `main`。

## 1. 分支模型

```text
main                        稳定基座：baseline + 统一实验设施
 ├── idea/<short-name>      一个 idea 一个分支
 ├── idea/<another-idea>    并行探索，互不干扰
 └── chore/<topic>          工程维护（文档、工具、测试）
```

规则：

- `main` 上只合并已经验证、可复现、测试通过的改动。
- 每个 idea 从最新的 `main` 签出，分支名使用小写字母、数字和下划线：
  `idea/dual_domain_transformer`、`idea/quantized_feedback_v2`。
- 一个分支只做一件事。不要把两个独立 idea 混在同一分支。
- 实验产物（`experiments/`、`data/`、`*.pth`）不进 git。

```bash
git switch main
git pull
git switch -c idea/<short-name>
```

## 2. Idea 生命周期

### 2.1 写清假设

在 `docs/ideas/<short-name>.md` 记录（可从
[`ideas/TEMPLATE.md`](ideas/TEMPLATE.md) 复制）：

- 要解决的问题 / 观察到的 baseline 空白
- 方法核心与关键改动
- 预期收益与可能的风险
- 消融计划

先写假设，再写代码。这样即使实验失败，也能留下可复用的结论。

### 2.2 实现

1. 在 `csifeedback/models/<short-name>.py` 实现网络，继承 `CSIAutoencoder`。
2. 用 `@register_model("<short-name>")` 注册工厂函数。
3. 在 `csifeedback/models/__init__.py` 导入该模块。
4. 在 `csifeedback/configs/ideas/<short-name>/` 下添加配置。

如果使用标准 MSE + 单优化器（Adam / AdamW / SGD）+ const / cosine 调度，
不需要写 trainer，`DefaultTrainer` 会接管。需要自定义 loss、双优化器或特殊
更新逻辑时，才新增专用 trainer。

### 2.3 快速验证

```bash
# 1. 全量单元测试
pytest tests/ -q

# 2. 2 个 epoch 的 smoke test
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/<short-name>/<short-name>_indoor_cr4.yaml \
    -o training.epochs=2 training.checkpoint_freq=0

# 3. 检查产物
ls experiments/<short-name>/indoor_cr4/<timestamp>/
cat experiments/<short-name>/indoor_cr4/<timestamp>/metrics.json
```

smoke test 要确认：

- 前向 shape 正确，loss 能下降；
- `metrics.json` 正常生成；
- 没有 NaN / Inf；
- GPU 显存与速度可接受。

### 2.4 完整实验

至少覆盖目标场景与目标压缩比。典型顺序：

1. 先跑一个代表性配置（如 indoor CR=1/4）验证方法有效；
2. 再扩展到完整 CR 矩阵；
3. 最后跑 outdoor 或第二个场景验证泛化。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/<short-name>/<short-name>_indoor_cr4.yaml
```

长跑建议放在 `tmux` / `screen` 中，日志重定向到文件：

```bash
tmux new -s <short-name>
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/<short-name>/<short-name>_indoor_cr4.yaml \
    2>&1 | tee experiments/<short-name>_indoor_cr4.log
```

### 2.5 与 baseline 对比

```bash
# 每个 (model, scenario, CR) 只保留完整运行，按 NMSE 排序
python scripts/compare.py --latest --min-epochs 1000

# Markdown，方便贴进实验记录
python scripts/compare.py --latest --min-epochs 1000 --format markdown
```

公平性检查：

- 同一 `data/` 文件与同一 train/val/test 划分；
- 同一 batch size 或至少同一 epoch 预算；
- 同一指标实现；
- 同样的 checkpoint 选择策略（test rho / test NMSE / val loss）；
- 对比时使用完整运行，不用 smoke test 或中途结果。

新 idea 推荐在配置中设置：

```yaml
training:
  select_best: "val_loss"
```

这样只用验证集选择 checkpoint，测试集只用于最终报告。baseline 参考表默认使用
`select_best: "all"`（按 test rho / test NMSE 保存）。如果新模型与 baseline 的
选择策略不同，必须在结果中注明；严格对比时建议用相同策略重跑 baseline。

### 2.6 多随机种子

单次结果不足以支撑“SOTA”结论。建议对关键配置至少跑 3 个种子：

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
      --config csifeedback/configs/ideas/<short-name>/<short-name>_indoor_cr4.yaml \
      -o training.seed=$seed
done
```

报告时给出均值 ± 标准差；如果只跑单次，明确标注。

## 3. 命名与目录规范

| 对象 | 规范 | 示例 |
|------|------|------|
| 分支 | `idea/<short_name>` | `idea/dual_domain_transformer` |
| 模型名 | 小写 + 下划线 | `dual_domain_transformer` |
| 模型文件 | `csifeedback/models/<name>.py` | `dual_domain_transformer.py` |
| 配置目录 | `csifeedback/configs/ideas/<name>/` | `.../ideas/dual_domain_transformer/` |
| 配置文件 | `<name>_<scenario>_cr<X>.yaml` | `dual_domain_transformer_indoor_cr4.yaml` |
| idea 记录 | `docs/ideas/<name>.md` | `docs/ideas/dual_domain_transformer.md` |
| 实验目录 | 由训练器自动生成 | `experiments/<name>/indoor_cr4/<timestamp>/` |

## 4. 合并回 main 的标准

一个 idea 只有在满足以下条件时才合并：

- [ ] 代码通过 `pytest tests/`
- [ ] 至少一个代表性配置完整训练（1000 epoch）并超过对应 baseline
- [ ] 关键配置有 3 个种子的均值 ± 标准差（或明确说明单次结果）
- [ ] 与 baseline 使用完全相同的数据、指标和评测流程
- [ ] 没有修改 baseline 代码（如有，已重跑全部 baseline 并更新文档）
- [ ] 新增测试覆盖模型实例化、前向 shape、单 batch 过拟合
- [ ] `docs/ideas/<name>.md` 写清方法、结果、消融与局限性
- [ ] README / docs 中的结果表已更新
- [ ] 没有提交 `experiments/`、`data/`、检查点或日志

合并方式：squash merge 或普通 merge 均可，但合并后的 `main` 必须保持测试通过。

## 5. 负结果与放弃的 idea

负结果同样有价值。不要直接删除分支；在 `docs/ideas/<name>.md` 中记录：

- 做了什么、为什么没 work；
- 哪些配置跑过、结果如何；
- 可能的原因与后续建议。

如果确定不再继续，可以保留分支不合并，或在团队同意后删除。

## 6. 工程维护

不属于具体 idea 的改动走 `chore/<topic>` 分支，例如：

- 新增实验对比工具；
- 改进日志或指标落盘；
- 更新文档；
- 修复测试。

工程改动同样需要 `pytest tests/` 通过，并且不能改变 baseline 的数值行为。
