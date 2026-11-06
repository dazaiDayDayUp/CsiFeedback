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
| `axis_codec_indoor_cr64_1000ep_seed0.json` | `experiments/axis_codec/indoor_cr64/20261106_112653` | CR1/64，1000 epoch，seed 0，复评 -8.942 dB / rho 0.9349 |
| `axis_codec_indoor_cr8_1000ep_seed0.json` | `experiments/axis_codec/indoor_cr8/20261106_144626` | CR1/8，1000 epoch，seed 0，复评 -20.805 dB / rho 0.9931 |
| `axis_codec_indoor_cr32_1000ep_seed0.json` | `experiments/axis_codec/indoor_cr32/20261106_174101` | CR1/32，1000 epoch，seed 0，复评 -11.485 dB / rho 0.9655 |
| `axis_codec_indoor_cr4_mse_300ep_seed0.json` | `experiments/axis_codec/indoor_cr4/20261106_054746` | 消融：MSE 目标，300 epoch，-27.485 dB |
| `axis_codec_indoor_cr4_nobase_300ep_seed0.json` | `experiments/axis_codec/indoor_cr4/20261106_072951` | 消融：去线性基座，300 epoch，-28.549 dB |
| `axis_codec_indoor_cr4_densemix_300ep_seed0.json` | `experiments/axis_codec/indoor_cr4/20261106_064004` | 消融：稠密角度混合，300 epoch，-28.736 dB |
| `baseline_reeval.json` | `paper/scripts/eval_baselines.py` | 三个 baseline × 5 个 CR 的固定 checkpoint 复评（NMSE/rho） |

论文图表由 `paper/scripts/data_prep.py` 读取本目录（实时实验目录存在时优先使用实时
结果，否则用这里的归档），再生成 `paper/data/meta.json` 与全部插图。
