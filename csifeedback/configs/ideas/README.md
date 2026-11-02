# 研究 idea 配置

每个新 idea 在独立 feature 分支上工作，配置放在本目录下按 idea 命名的子目录：

```
csifeedback/configs/ideas/<idea>/
├── <idea>_indoor_cr4.yaml
├── <idea>_indoor_cr16.yaml
└── ...
```

约定：

- `<idea>` 使用小写字母、数字和下划线，例如 `dual_domain_transformer`。
- 文件名格式 `<idea>_<scenario>_cr<X>.yaml`，`scenario` 取 `indoor` / `outdoor`，
  `cr` 取 `4`、`8`、`16`、`32`、`64`。
- baseline 配置位于 `../baselines/`，只读参考，不要在 idea 分支中修改。
- 新增配置后会自动被 `tests/test_configs.py` 与 `scripts/train_all.py` 发现。

示例（最小配置）：

```yaml
model:
  name: "my_idea"
  encoded_dim: 512
  extra:
    depth: 2

data:
  data_dir: "./data"
  scenario: "indoor"
  batch_size: 200

optimizer:
  name: "adam"
  lr: 1.0e-3
  betas: [0.9, 0.999]

scheduler:
  name: "cosine"

training:
  epochs: 1000
  checkpoint_dir: "./experiments"
  select_best: "val_loss"   # 新 idea 推荐，避免用测试集选模型
```
