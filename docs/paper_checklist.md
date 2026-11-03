# 论文产出清单

本文档把“做出新 SOTA”拆解为可检查的论文级证据。每个 idea 在准备投稿前，
逐项确认。

## 1. 研究问题与贡献

- [ ] 明确指出 baseline 在什么场景、什么压缩比下存在什么不足
- [ ] 用一句话概括核心假设
- [ ] 列出 2–3 条可验证的贡献（方法、机制、实验发现）
- [ ] 方法图、关键公式、符号表
- [ ] 与最接近的已有工作做区别说明

## 2. 公平对比

- [ ] 使用 `data/` 中同一份 COST2100 数据与同一 train/val/test 划分
- [ ] 使用相同的 `compute_nmse` / `compute_rho` 实现
- [ ] 与 baseline 使用相同的 epoch 预算和可比的 batch size
- [ ] 明确 checkpoint 选择策略（`training.select_best`）
- [ ] 报告 `metrics.json` 的 `selected`（实际选中 checkpoint 的指标），并与 `best` 区分
- [ ] 新模型与 baseline 的选择策略不同时，在论文中注明
- [ ] 不使用测试集调参；测试集只用于最终报告

## 3. 实验矩阵

- [ ] indoor / outdoor 两个场景
- [ ] CR = 1/4、1/8、1/16、1/32、1/64
- [ ] 至少一个场景给出完整 CR 曲线
- [ ] 与 CLNet、STNet、CsiNet 的 NMSE / rho 对比表
- [ ] 关键配置优于最强 baseline，且提升幅度有统计意义

## 4. 随机种子与稳定性

- [ ] 关键配置至少 3 个随机种子（如 0、1、2）
- [ ] 报告均值 ± 标准差
- [ ] 说明均值和标准差的计算方式
- [ ] 如果只跑单次，明确标注原因

## 5. 消融实验

- [ ] 去掉核心模块
- [ ] 替换核心模块为简单基线（如 identity / 单分支）
- [ ] 关键超参数（depth、heads、latent 结构等）的敏感性
- [ ] 训练策略（loss、scheduler、select_best）的影响
- [ ] 每个消融只改变一个因素

## 6. 效率与复杂度

- [ ] 参数量
- [ ] FLOPs 或 MACs
- [ ] 训练显存与单 epoch 时间
- [ ] 推理延迟（如与 CSI 反馈时延相关）
- [ ] 与 baseline 的复杂度对比表

## 7. 结果分析

- [ ] 不同 CR 下的性能趋势解释
- [ ] indoor / outdoor 差异解释
- [ ] 失败案例或性能下降场景
- [ ] 可视化（如重建 CSI、注意力图、误差图）
- [ ] 方法局限性

## 8. 可复现性

- [ ] 配置文件在 `csifeedback/configs/ideas/<idea>/`
- [ ] 运行命令写入 `docs/ideas/<idea>.md`
- [ ] `metrics.json` 记录 git commit、配置、最优指标和评估记录
- [ ] 论文中报告的结果能由对应 commit + config 重跑
- [ ] 检查点、日志、实验目录未提交到 git

## 9. 写作与发布

- [ ] 方法、实验、消融、复杂度四部分齐全
- [ ] 图表编号与正文引用一致
- [ ] 复现实验命令与超参数表放入附录
- [ ] 代码整理为可公开状态（无本地路径、无个人信息、无大数据文件）
- [ ] README 与 `docs/baselines.md` 的结果表已更新
