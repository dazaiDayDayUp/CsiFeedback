# CSI 反馈文献调研

本目录用于沉淀 Massive MIMO CSI 反馈方向的文献调研，服务于本仓库的 baseline 对比与后续研究。
**当前阶段只做调研，不提出新方法或新 idea。**

> 状态：2026-11-03 完成第一轮调研。8 个主题分册 + 1 个总览，合计 166 条参考文献条目
> （跨册有重复）。2025–2026 年条目含预印本，各分册已标注核实程度。数值均为文献报告值，
> 未在本仓库重新训练或复评。

## 调研范围

| 项目 | 设定 |
|------|------|
| 任务 | Massive MIMO 下行 CSI 压缩与重建（autoencoder-based CSI feedback） |
| 场景 | COST2100 indoor / outdoor；FDD 下行；低速与高移动场景 |
| 压缩比 | 1/4、1/8、1/16、1/32、1/64（瓶颈维度 2048/CR） |
| 指标 | NMSE（dB）、rho（余弦相似度） |
| 基线 | CsiNet、CLNet、STNet（本仓库的固定参照） |
| 时间范围 | 2018 年 CsiNet 起，重点覆盖近 5 年及可核实的最新工作 |

相邻但相关的方向（3GPP CSI 码本、AI/ML 空口标准化、物理层自编码器、联合导频/反馈）
只在与 CSI 反馈的对比或评测协议直接相关时纳入。

## 方法

- 来源优先级：论文原文（arXiv / DOI / 官方页面）> 作者项目页 > 综述 > 二手博客。
- 每篇文献至少核实标题、第一作者、年份与可访问链接；venue 无法确认时标注「未核实」。
- 数值（NMSE / rho / 参数量 / 复杂度）只在原文或摘要中确实出现时记录，并注明数据集、
  场景、CR、epoch 预算与 checkpoint 选择方式；不把不同协议的数值直接横向比较。
- 调研笔记中的判断与原文事实分开写：事实用陈述句并给出引用，推断标注「本调研判断」。
- 不提交 PDF、数据集或大文件；只保留 Markdown 笔记与链接。

## 文件索引

| 文件 | 内容 |
|------|------|
| [`01_classic_autoencoders.md`](01_classic_autoencoders.md) | 2018-2021 经典自编码器/卷积架构与评测约定 |
| [`02_attention_transformers.md`](02_attention_transformers.md) | 注意力与 Transformer 类方法 |
| [`03_complex_lightweight.md`](03_complex_lightweight.md) | 复数神经网络、轻量化与高效推理 |
| [`04_model_driven_priors.md`](04_model_driven_priors.md) | 模型驱动、压缩感知与物理先验 |
| [`05_generative_foundation.md`](05_generative_foundation.md) | 生成式模型、自监督与基础模型 |
| [`06_temporal_mobility_overhead.md`](06_temporal_mobility_overhead.md) | 时变信道、反馈开销/延迟与可变压缩比 |
| [`07_standards_adjacent.md`](07_standards_adjacent.md) | 3GPP 码本、AI/ML 空口与相邻任务 |
| [`08_reported_results.md`](08_reported_results.md) | 跨论文数值汇总与可复现性差异 |
| [`09_synthesis.md`](09_synthesis.md) | 分类、时间线与文献明确指出的开放问题 |

## 维护约定

- 新增调研先更新本索引，再新增文件；文件名使用两位序号 + 短横线小写主题。
- 引用格式：`[序号] Author et al. Title. Venue, Year. URL`，每个文件文末自包含参考文献。
- 修改已有结论时保留「核查记录」，写明依据来自原文、摘要还是二手来源。
- 本目录不存放实验产物；实验记录仍在 `docs/ideas/<name>.md` 与 `experiments/`。
