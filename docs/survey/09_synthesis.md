# 调研总览：分类、时间线与可比性

> 汇总日期：2026-11-03。本文件只归纳八个分册的文献事实与协议差异，不提出新方法或新 idea。
> 各分册合计 166 条参考文献条目（跨册有重复引用）；2025–2026 年条目中相当一部分是预印本，
> 分册内已逐条标注「未核实」或「摘要级」。

## 1. 研究版图

CSI 反馈的文献可以按「压缩链路中改进发生在哪一层」分成七条路线。它们不是互斥的；
2024 年后的工作越来越常把两条以上路线组合。

| 路线 | 核心问题 | 代表工作 | 与本仓库设定的距离 |
|------|----------|----------|--------------------|
| 经典卷积自编码器 | 用端到端网络替代 CS 迭代重建 | CsiNet、CsiNet+、DeepCMC、CRNet | 最近；本仓库三个 baseline 的直接来源 |
| 注意力 / Transformer | 用全局或窗口注意力建模角-时延相关 | Attention-CsiNet、DS-NLCsiNet、TransNet、STNet、CSIFormer、SwinCFNet | 近；STNet 已是本仓库 baseline |
| 复数与轻量化 | 保留幅度-相位结构、降低 UE 侧开销 | CLNet、CV-3DCNN、ENet、ShuffleCsiNet、BCsiNet、NTC、STQENet | 近；CLNet 已是 baseline |
| 模型驱动与先验 | 把稀疏、低秩、多径、统计先验写进链路 | FISTA-Net、LORA、L2O、LRP、HS-PINNnet、PM-VQ、CSI-PPPNet | 中；常改反馈表示或推理方式 |
| 生成式与基础模型 | 用分布先验补足低码率下的欠定信息 | GAN/VAE、扩散解码、CsiCoGen、WiFo、HeterCSI、LLM4CSI | 中远；多按比特预算或跨配置评测 |
| 时变与开销 | 建模时间相关、延迟、反馈周期与可变率 | CsiNet-LSTM、MarkovNet、CSILaBS、DCA、PM-VQ | 远；本仓库数据无帧维度 |
| 标准化与相邻任务 | 与码本、预编码、系统级 KPI 对齐 | 3GPP Type I/II、eType II、TR 38.843、物理层自编码器、联合导频 | 远；反馈对象和指标不同 |

各分册的详细分类与代表工作：

- [`01_classic_autoencoders.md`](01_classic_autoencoders.md)：CsiNet 范式的形成，以及 CRNet 把训练配方变成结果的一部分。
- [`02_attention_transformers.md`](02_attention_transformers.md)：注意力作用域从特征重标定、全注意力、窗口/可分离到层级化与跨流。
- [`03_complex_lightweight.md`](03_complex_lightweight.md)：复值建模、结构轻量化、量化/剪枝/蒸馏，以及复杂度口径的异质性。
- [`04_model_driven_priors.md`](04_model_driven_priors.md)：CS、展开网络、低秩/多径参数化、统计先验、未训练与 PnP。
- [`05_generative_foundation.md`](05_generative_foundation.md)：GAN/VAE、扩散/score-based、自监督预训练与无线基础模型。
- [`06_temporal_mobility_overhead.md`](06_temporal_mobility_overhead.md)：单快照与序列两类设定，以及延迟、可变率、互易性和多用户。
- [`07_standards_adjacent.md`](07_standards_adjacent.md)：3GPP 码本、AI/ML 空口、物理层自编码器与联合导频/反馈。
- [`08_reported_results.md`](08_reported_results.md)：可核实的 NMSE/rho/复杂度数值与文献自述的评测不一致。

## 2. 时间线与五次重心转移

| 阶段 | 主要变化 | 节点工作 |
|------|----------|----------|
| 2018–2019 | 端到端自编码器替代 CS 迭代；开始引入时域与量化 | CsiNet、CsiNet-LSTM、CsiNet+ |
| 2020–2021 | 全卷积/率失真、多分辨率、复数输入与轻量化；训练配方被显式讨论 | DeepCMC、CRNet、CLNet |
| 2021–2022 | 注意力与 Transformer 成为独立分支；模型驱动展开网络集中出现 | TransNet、STNet、CSIFormer、FISTA-Net、LORA |
| 2023–2024 | 生成式先验进入解码端；量化/熵编码与端到端训练结合；基础模型开始预训练 | 扩散信道估计、SwinCFNet、NTC、LWM |
| 2025–2026 | 基础模型与可变比特率生成式反馈并行；评测可复现性被单独提出 | WiFo-2、HeterCSI、CsiCoGen、PM-VQ、CFM-Bench |

文献中可直接读到的五次重心转移：

1. **从「更好的回归器」到「更好的先验」。** 2018–2021 年主要改编码器结构与量化损失；
   2023 年后更多工作把低压缩比重建明确为欠定问题，改进点转向解码端使用什么分布或结构先验 [05](05_generative_foundation.md)、[04](04_model_driven_priors.md)。
2. **从固定压缩比浮点码字到比特预算/可变长度。** CsiNet 用 `M/(2N_tN_c)` 定义压缩比；
   后续出现 BPP、固定 bit、码本前缀长度等口径 [08](08_reported_results.md)、[03](03_complex_lightweight.md)。
3. **从单快照到序列。** CsiNet-LSTM 之后，时间相关性从「附加增益」变成结构性假设；
   反馈延迟也从误差项变成设计对象 [06](06_temporal_mobility_overhead.md)。
4. **从单场景到跨配置/跨场景。** 2024 年后出现多速率、多维度、多场景统一预训练；
   但泛化结论多来自作者自报，跨仿真器迁移仍被列为障碍 [05](05_generative_foundation.md)、[08](08_reported_results.md)。
5. **从单一 NMSE 到任务导向与系统级指标。** 波束成形增益、和速率、BLER 与 GCS 开始与 NMSE/rho 并列；
   文献明确指出 NMSE 只是代理指标 [01](01_classic_autoencoders.md)、[07](07_standards_adjacent.md)。

## 3. 与本仓库设定的对齐关系

本仓库当前设定是：COST2100 indoor/outdoor、`(2,32,32)` 单快照、压缩比 1/4–1/64、
浮点瓶颈（默认不量化）、NMSE + rho、两侧联合训练、baseline 用测试集指标选模型
（`select_best: "all"`）。

据此可以把文献分成三档：

**A. 可直接对齐。** 使用同一批 COST2100 MAT 文件、32×32 截断、100k/30k/20k 划分、
固定 CR 浮点码字、只报 NMSE 的卷积/注意力/轻量化工作。CsiNet、CRNet、CLNet、STNet、
TransNet、SwinCFNet 等属于这一档。即便如此，训练预算（1000 vs 2500 epoch）与
checkpoint 选择仍常不同，横向数字只能作为量级参考。

**B. 需要补条件才能对齐。** 量化/熵编码、可变码率、BPP 报告、多速率单模型、
任务导向损失。它们改的是「反馈多少比特」或「优化什么目标」，不是同一个 CR 网格；
引用时必须同时写明比特预算、是否量化、指标定义 [03](03_complex_lightweight.md)、[06](06_temporal_mobility_overhead.md)。

**C. 不可直接对齐。** 时变序列、非 COST2100 数据集（DeepMIMO、Sionna/3GPP、QuaDRiGa、
实测数据）、3GPP 码本/系统级评估、跨配置基础模型、物理层端到端自编码器。
它们的信道模型、反馈对象或指标与本仓库不同 [06](06_temporal_mobility_overhead.md)、
[07](07_standards_adjacent.md)、[05](05_generative_foundation.md)。

一个具体的对齐风险：本仓库 `docs/baselines.md` 的记录是训练中 `test_freq=10` 观察到的最优值，
且 rho 与 NMSE 可能来自不同 epoch；文献中的数值大多是「训练结束/验证最优」的单一报告。
两者不是同一种统计量，不能逐格比较 [08](08_reported_results.md)。

## 4. 文献中反复出现的开放问题

以下每条都至少有一个分册给出原文出处；这里只归类，不提出解决方向。

### 4.1 评测与可复现性

- 训练预算没有统一：CRNet 的 epoch 消融显示同一网络在 1000 与 2500 epoch 间可差 1.82 dB；
  CLNet 明确用 1000 epoch 复现的 CRNet 作对照，而 CRNet 原文是 2500 epoch [01](01_classic_autoencoders.md)、[08](08_reported_results.md)。
- checkpoint 选择大多未说明：除 CsiNet 明确写按验证损失选模型外，所核查论文多数没有交代
  报的是最后 epoch、验证最优还是测试最优 [08](08_reported_results.md)。
- CR 定义不统一：CsiNet 正文用 1024 个复数元素作分母，官方代码用 2048 个实数；
  CsiNet+ 后来写成 `2N_tN_c/M` [01](01_classic_autoencoders.md)。
- NMSE 计算域与反归一化不统一：角-时延域、是否减 0.5、逐样本还是 pooled，都会改变 dB 值 [01](01_classic_autoencoders.md)、[08](08_reported_results.md)。
- rho 的实现范围不统一：CsiNet 官方代码只取前 125 个子载波，而论文定义是对全部子载波平均；
  2019 年后多数工作只报 NMSE，rho 逐渐退出对照表 [08](08_reported_results.md)。
- 复杂度口径不统一：encoder-only 还是全网、fvcore 还是 thop、是否计 BN、理论 FLOPs 还是实测时延，
  都会改变「轻量」结论；已有第三方更正前人 FLOPs 的实例 [03](03_complex_lightweight.md)、[08](08_reported_results.md)。
- 基线未重跑：生成式与比特级工作常直接引用他人曲线，量化方式与编码方式不同，
  比较混合了架构差异与码率差异 [05](05_generative_foundation.md)。

### 4.2 低压缩比下的欠定重建

- 文献把重建歧义的处理分为：只改损失（GAN/VAE）、二阶统计先验补全、条件后验采样（扩散）、
  把歧义转成概率模型；这四类没有在同一协议下被比较过 [05](05_generative_foundation.md)。
- 扩散解码的计算复杂度仍是部署障碍；只有少数论文报告时延、FLOPs 或吞吐 [05](05_generative_foundation.md)。
- 生成随机性通常没有进入指标：NMSE/rho 多按单次或固定种子评估，未见多次采样的均值 ± 方差 [05](05_generative_foundation.md)。

### 4.3 量化、熵编码与浮点码字的断层

- 量化是否在环会改变 NMSE 量级：LORA 联合训练与分步训练的 NMSE 分别为 0.0053 与 0.0069 [04](04_model_driven_priors.md)。
- 全精度、固定比特、NMSE–BPP 三种报告方式并存，尚无一者成为默认 [08](08_reported_results.md)。
- 粗量化会形成失真地板：STQENet 观察到 2 bit 量化在高 SNR 下出现性能地板 [03](03_complex_lightweight.md)。
- TransNet+ 明确指出现有 Transformer 反馈未考虑码字量化 [02](02_attention_transformers.md)。

### 4.4 跨场景、跨配置与泛化

- 多数模型按单一场景训练，跨场景退化；AiANet、WiFo-CF、HeterCSI 都把跨配置/跨分布作为核心问题 [02](02_attention_transformers.md)、[05](05_generative_foundation.md)。
- 基础模型的增益可能来自预训练规模而非结构，训练预算与数据规模常不可比 [05](05_generative_foundation.md)。
- 跨仿真器零样本迁移会出现显著退化，联合性能-成本 KPI 下的 baseline 选择仍是落地障碍 [08](08_reported_results.md)。

### 4.5 时变、延迟与反馈周期

- 无记忆模型浪费帧间相关性，而简单的多帧拼接没有显式建模潜变量动力学 [06](06_temporal_mobility_overhead.md)。
- 时序方案增加参数量与计算量；RNN 的计算与存储成本被列为部署障碍 [06](06_temporal_mobility_overhead.md)。
- 固定压缩比难以适应非平稳信道，系统级需要动态选率 [06](06_temporal_mobility_overhead.md)。
- 反馈延迟与同步误差会造成层间干扰与符号间干扰，但现有分析多针对容量而非压缩反馈端到端链路 [06](06_temporal_mobility_overhead.md)。

### 4.6 UE 侧与 BS 侧的不对称

- CLNet 的总 FLOPs 下降主要来自 BS 侧解码器，UE 侧编码器反而略重 [03](03_complex_lightweight.md)。
- 展开网络与 PnP 常把大部分计算放在 BS 侧，UE 只保留测量矩阵或线性投影；
  「总参数量」会掩盖两端分布差异 [04](04_model_driven_priors.md)。
- 标准化讨论把 UE 侧轻量、访存与并行度列为独立约束，FLOPs 不能完全代表推理速度 [07](07_standards_adjacent.md)。

### 4.7 指标与任务目标错配

- NMSE/rho 是波束成形与系统吞吐的代理指标，二者非单调 [01](01_classic_autoencoders.md)、[07](07_standards_adjacent.md)。
- 任务导向损失（BF 增益、和速率、有效 CSI）会改变模型排序，不能与 NMSE 表直接互换 [01](01_classic_autoencoders.md)、[07](07_standards_adjacent.md)。

### 4.8 标准化与部署

- 3GPP 的 CSI 反馈是隐式、以比特计、受码本约束的；基于 COST2100 的结果不能直接与 Type I/II 码本比较 [07](07_standards_adjacent.md)。
- 双向模型的跨厂商训练（Type 1/2/3）、模型配对、生命周期管理与回退是标准化核心难点 [07](07_standards_adjacent.md)。
- 标准化只定义配置、验证、激活、监控与回退语义，不规定模型结构；这与学术论文的目标函数不同 [07](07_standards_adjacent.md)。

## 5. 读数值时必须注意的边界

1. **只记录「已报告值」，不是「可复现值」。** [`08_reported_results.md`](08_reported_results.md) 中的表来自原文表、作者仓库或明确标注的转引；
   本调研没有重新训练任何文献模型。
2. **CR 的分母可能是 1024 或 2048。** 同为「1/4」，码字可能是 256 个复数或 512 个实数。
3. **NMSE 的计算域可能是角-时延域或空间-频率域，且是否减 0.5 不统一。**
4. **rho 只在 CsiNet 原文中系统报告，且官方实现只用了 125 个子载波。**
5. **训练预算与 checkpoint 选择常不同。** 1000 vs 2500 epoch、验证最优 vs 测试最优 vs 最后 epoch，
   都可能造成数 dB 差异。
6. **复杂度数值至少三种口径。** encoder-only vs 全网；fvcore vs thop；FLOPs vs 能耗 vs BPP。
7. **2026 年条目多为预印本。** 分册内已标注；引用前应再次核对正式发表信息。

## 6. 命名澄清

调研中至少出现四组容易混淆的名称，后续引用时应按这里的口径：

- **DS-Net**：未检索到以该名发表的 CSI 反馈工作；最接近的是 DS-NLCsiNet（Yu et al., IEEE Communications Letters 2020，非局部网络）。
- **CSFormer / CSIFormer**：CSformer（Ye et al., TIP 2023）是图像压缩感知方法；CSIFormer（Bi et al., WCL 2022）才是 CSI 反馈的卷积-窗口 Transformer。
- **FeCoMP**：在 3GPP 语境中是 LTE Rel-15 的 further enhanced CoMP 工作项，不是 NR 码本名称；
  NR 多 TRP 相干联合传输更常见的命名是 eType2CJT / feType2CJT。
- **STNet 的年份**：arXiv 2022，IEEE Wireless Communications Letters 2023。分册中两种写法都出现过，指同一工作。

## 7. 结论

- 2018 年 CsiNet 建立的「COST2100 + 32×32 + 自编码器 + NMSE/rho」范式仍是学术比较的共同底座；
  2026 年的新工作仍以它为公共参照。
- 网络结构的主线从卷积、注意力、复数/轻量，扩展到模型驱动先验、生成式解码与基础模型；
  但**评测协议没有同步统一**，这是跨论文比较最大的障碍。
- 固定压缩比浮点码字、比特预算、可变长度码字是三种不同的反馈接口；本仓库目前属于第一种。
- 文献中增长最快的两条线是「生成式/基础模型」与「比特级/可变率」；两者都把评测从单一 NMSE
  推向比特、时延、泛化或系统 KPI。
- 与本仓库设定最接近、且可复现性信息最完整的仍是 2018–2022 年的 CsiNet/CRNet/CLNet/STNet 一线；
  后续工作的比较必须先补协议字段，再谈数值高低。
