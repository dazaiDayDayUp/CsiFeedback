# 投稿前自审记录（Paper Self-Review）

对应 `PAPER_PLAN.md` §6 的自审清单。每项给出检查方式与证据，便于复核。

## 1. 标题 / 作者 / 摘要 / 关键词

- 标题：*AxisCodec: Geometry-Aware CSI Feedback on the Angular–Delay Cylinder*（PDF 第 1 页）。
- 作者：`Ziheng Wang and Kimi Code`，脚注含 `666zihengwang@gmail.com` 与稿件说明（PDF 第 1 页左下）。
- 摘要：约 230 词，位于 150–250 词区间；包含问题、诊断结论、方法与关键数字。
- 关键词：5 个（CSI feedback, massive MIMO, angular–delay domain, physics-informed
  learning, rate–distortion）。
- PDF 元数据（Title/Author/Keywords）由 `\hypersetup` 写入，`pdfinfo` 可查。

## 2. 交叉引用与编号

- `pdftotext main.pdf - | grep -c "??"` → **0**。
- 图 6 幅、表 5 张、公式 4 个、章节 7 节，编号连续；`\label`/`\ref` 一一对应
  （脚本核对 labels 与 refs 集合）。
- 参考文献 22 条，正文 22 条全部被引用（脚本核对 `\cite` 与 `\bibitem` 集合）。

## 3. 公式与记号

- (1) Dirichlet 核与角度–时延坐标；(2) 循环角度混合（DFT 域滤波）；
  (3) 线性基座 + 非线性残差解码；(4) 逐用户 NMSE。
- (3)(4) 记号统一：`Ĥ = Pz + g_θ(z)`、`x̂ = 0.5 + Ĥ`，式 (4) 用 `x̂_i, x_i`。
- dB 定义统一为 `10 log10(·)`，NMSE 越低越好、ρ 越高越好，全文一致。

## 4. 图（逐张渲染-读图，150–300 dpi）

| 图 | 检查点 | 结论 |
|---|---|---|
| Fig.1 几何 | 极坐标径向标度真实（12% 峰值不再被截断）、31|0 接缝可见、色标与双轴刻度清晰 | 通过 |
| Fig.2 误差份额 | `step="mid"` 半格错位已修正；signal share 进图例；head/tail 分界清楚 | 通过 |
| Fig.3 架构 | 角度→时延流程有箭头；LN/GELU 归属明确；线性基座/残差汇入“+”清楚 | 通过 |
| Fig.4 率失真 | 增益符号修正为 +；增益行移至轴下；PCA 末点不再被裁切 | 通过 |
| Fig.5 定性 | PowerNorm 使弱时延结构可见；行标签在面板内不与 ylabel 重叠 | 通过 |
| Fig.6 训练 | 三种子带状范围；300-epoch 标注与图例不重叠 | 通过 |

- 配色为 Okabe–Ito 变体（色盲友好）；字体全为 STIX/Type-42 嵌入。
- 图内最小字号：主刻度 ≥7.2 pt，个别辅助标注 6.8–7.0 pt（IEEE 允许 6–7 pt 辅助文字）。

## 5. 表

- 全部为 booktabs 三线表，caption 在上，最优值加粗，单位写在 caption 或表头。
- Table I 的 `reals / sparse k / reals` 两组列由 caption 说明其含义。
- Table II 的 baseline 数值来自固定 checkpoint 复评（`paper/scripts/eval_baselines.py`
  → `docs/ideas/metrics/baseline_reeval.json`）。
- Table III/IV/V 的数值可由 `paper/scripts/ablation_table.py` 与 `paper/data/meta.json` 复核。

## 6. 参考文献

- 22 条，IEEE 格式；期刊有卷/期/页，会议有地点与页码，arXiv 有编号。
- 全部条目在正文中被引用；无 “to appear” 之外的未完成条目。

## 7. 语言

- 无 AI 高频词（`novel/leverage/delve/crucial/furthermore/moreover/state-of-the-art` 等
  全文检索为 0 命中）。
- 段落以事实与数字驱动；Limitations 与 Reproducibility 各一段，不过度防御。

## 8. 数值一致性

正文、表、图三处关键数字逐一核对（脚本/单元计算）：

| 陈述 | 数值 | 来源 |
|---|---|---|
| CR1/4 三 seed 均值±std | -32.52 ± 0.55 dB | `axis_codec_indoor_cr4_1000ep_seed{0,1,2}.json` |
| vs STNet | +2.34 dB（4.2× 种子 std，6.0× run 内 std 0.387） | 同上 + `baseline_reeval.json` |
| 头部/尾部误差（CR1/4） | 3.7e-4 / 9.4e-3（STNet），3.0e-5 / 7.6e-3（ours） | `meta.json: decomp_*` |
| taps 7–9 误差份额 / 信号份额 | 42% / 0.7% | 同上（0.4196 / 0.0070） |
| head→tail 岭回归 R² | 0.235（正文写 23%） | 40k 训练样本独立复算 |
| 消融四分位增益 / 改善样本 | 1.42–1.60 dB / 97.5% | 300-epoch checkpoint 复算 |
| PCA-512 误差份额 | 0.3%（taps 0–3）/ 99.7%（4–31） | `experiments/pca/indoor_512.pt` 复算 |
| 参数 / FLOPs | 2.13 M / 0.022 G（STNet 2.13 M 有效 / 0.014 G） | `analysis/complexity.json` |

## 9. 复现性

- 图表全部由 `paper/scripts/*.py` 从 `paper/data/` 与研究目录/归档的 `metrics.json`
  生成；`data_prep.py` 不再依赖仓库外路径（analysis JSON 已随仓库保存）。
- 训练配置在 `csifeedback/configs/ideas/cylinder_codec/`；运行摘要归档在
  `docs/ideas/metrics/`。
- 论文第 VI 节含 Reproducibility 段落，说明配置、指标记录与单卡训练时长。

## 10. 版面

- `pdfinfo`：6 页；每页正文行均顶到版心底边（第 6 页右栏 98%）。
- 无 overfull hbox 告警；`\emergencystretch` 已设，个别 underfull 为正常段落末行。
- 字体全部嵌入（`pdffonts` 全为 `emb yes`）。
