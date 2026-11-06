# 关键运行的 metrics.json 存档

仓库策略不提交 `experiments/`（含检查点与日志），但论文需要可追溯的实验摘要，
因此把**关键运行**的 `metrics.json` 原文复制到这里。文件内包含：配置、`select_best`、
逐 epoch 的 val/test 记录、实际被选 checkpoint 的指标（`selected`）、耗时、参数量、
git commit。

| 文件 | 运行目录（worktree `cylinder_codec`） | 说明 |
|---|---|---|
| `axis_codec_indoor_cr4_300ep_seed0.json` | `experiments/axis_codec/indoor_cr4/20261105_172649` | CR1/4，300 epoch，seed 0，复评 -28.984 dB |
| `axis_codec_indoor_cr4_1000ep_seed0.json` | `experiments/axis_codec/indoor_cr4/20261105_182644` | CR1/4，1000 epoch，seed 0，复评 -32.462 dB |
| `axis_codec_indoor_cr4_1000ep_seed1.json` | `experiments/axis_codec/indoor_cr4/20261105_235526` | CR1/4，1000 epoch，seed 1，复评 -33.093 dB |
| `axis_codec_indoor_cr4_1000ep_seed2.json` | `experiments/axis_codec/indoor_cr4/20261106_025326` | CR1/4，1000 epoch，seed 2，复评 -31.995 dB |
| `axis_codec_indoor_cr16_1000ep_seed0.json` | `experiments/axis_codec/indoor_cr16/20261106_082212` | CR1/16，1000 epoch，seed 0，复评 -15.300 dB |
