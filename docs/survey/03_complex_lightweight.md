# 复数神经网络与轻量化 CSI 反馈

> 范围：FDD 下行 CSI 压缩与重建，输入 `(N, 2, 32, 32)`，压缩比 1/4–1/64，COST2100
> indoor/outdoor，指标 NMSE(dB) 与 rho。本文只梳理文献事实，不提出方法或改进方案。

## 1 范围、检索与写作约定

本文覆盖三条相互交叠的线索：（1）复数卷积、复数激活与复值 CSI 建模；（2）模型级轻量化，
包括轻量结构、二值化/量化、剪枝、知识蒸馏与低秩手段；（3）轻量化论文对复杂度（参数量、
FLOPs、时延、存储）的报告口径。相邻的时变预测、联合信源信道编码与 3GPP 码本只在直接
影响上述线索时纳入。

检索于 2026-11-03 进行，来源为 arXiv 题名/摘要页与 HTML 全文、OpenAlex 与 Semantic Scholar
的元数据、IEEE DOI 落地页和 IET 数字图书馆。检索词包括 `complex-valued CSI feedback`、
`complex convolution CSI feedback`、`lightweight CSI feedback`、`CSI feedback quantization`、
`CSI feedback pruning`、`knowledge distillation CSI feedback`、`low-rank CSI feedback`、
`efficient CSI feedback`，时间范围 2018-01 至 2026-11，优先 2021 年后的工作。只保留能核实
题名、第一作者、年份与可访问链接的条目；数值仅在原文全文或摘要中确实出现时记录。

一个贯穿全文的约定：不同论文的 COST2100 划分、epoch 预算、checkpoint 选择与复杂度统计边界
并不一致，因此本文不做跨论文的排名式比较，只在同一论文内部引用其自报结论。

## 2 分类与时间线

2018 年 CsiNet 确立“编码器在 UE、解码器在 BS”的自编码器范式后，轻量化大致沿三条路径展开：
结构替换（卷积分解、通道混洗、可分离注意力）、训练策略（知识蒸馏）、以及权重或码字的离散化
（二值化、量化、剪枝）。复值建模则是一条相对独立的分支，试图在实数神经网络中保留 CSI 的
幅度—相位结构。2025–2026 年的新工作更多以“码率—失真”或“参数效率”为组织方式，而不是单纯
追求固定压缩比下的 NMSE。

| 类别 | 代表工作（年份） | 主要手段 | 原文报告的主要开销指标 |
|---|---|---|---|
| 复值与实虚联合 | CV-3DCNN (2021)、CLNet (2021)、ENet (2021)、CVLNet (2022)、Liu et al. (2022) | 复值卷积、复值输入嵌入、实虚部复用 | FLOPs、参数量、精度百分比 |
| 轻量结构 | ShuffleCsiNet (2021)、STNet (2023)、SSE-Net (2026) | 深度可分离卷积、通道混洗、空间可分离注意力 | 参数量、FLOPs、“资源量” |
| 离散化与码字 | JC-ResNet (2020)、BCsiNet (2021)、ACRNet (2021)、Shape-Gain VQ (2024)、NTC (2025)、STQENet (2025) | 二值化、量化、熵编码、可变码率 | 内存、推理加速、BPP–NMSE |
| 剪枝与通用压缩 | Guo et al. (2020)、Erak & Abou-Zeid (2023)、SNIP-CsiNet (2023) | 剪枝、训练后量化、权重聚类 | 模型体积、实测时延 |
| 知识蒸馏 | Tang et al. (2021)、Cui et al. (2024) | 教师–学生、encoder-only KD | FLOPs 比例、推理时间比例 |
| 低秩 | TSLR (2022)、LRP (2026) | 信道低秩先验、秩一感知 | 复杂度的定性描述 |

## 3 复数神经网络

### 3.1 方法学源头

Trabelsi et al. 的 Deep Complex Networks 提供了复数卷积、复数批归一化与复值权重初始化的
基本构件，并把它们用于卷积前馈网络与卷积 LSTM；作者报告复值模型与对应的实值模型具有竞争力，
并在音频相关任务上取得当时的最好结果 [1]。这篇工作不是通信论文，但它是后续 CSI 复值建模
反复引用的方法基础。

### 3.2 CSI 反馈中的复值设计

CV-3DCNN 将复数三维卷积用于 FDD massive MIMO 的下行 CSI 预测，作者称其在复数域直接处理
CSI 矩阵以减少数据预处理带来的信息损失，摘要报告预测精度提升约 6 dB [2]。需要注意这是
**预测**任务，不是压缩反馈，压缩比与 NMSE 口径都与本文关注的设定不同。

CLNet 是复值线索上较有代表性的轻量化工作。它没有采用完整的复数卷积核，而是提出“forged
complex-valued input”：用 1×1 逐点卷积分别作用于实部与虚部通道，使同一复系数的实虚比（即
相位）在加权后仍被保留，再以 SE 模块做通道注意力；编码器另加 CBAM 空间注意力，解码器把
CRBlock 的 1×9 卷积缩为 1×3，并用 hard-sigmoid 替代 sigmoid 以减少指数运算 [3]。原文明确
指出复数核难以通过反向传播优化、训练困难且复杂度高，这是它选择“伪复值输入”而非真复值
网络的理由。

其实验设定为 COST2100（100,000/30,000/20,000 划分）、batch 200、1000 epoch、固定随机种子、
单卡 2080 Ti。原文报告相对 CRNet 平均 5.41% 的精度提升与平均 24.1% 的 FLOPs 下降；分压缩比
看，FLOPs 在 1/64、1/32、1/16、1/8、1/4 上分别少 18.00%、22.35%、25.20%、26.50%、28.36%
[3]。500 epoch 的消融显示，仅把首层换成 1×1 卷积即可相对 CRNet 平均提升 10.964%。但原文
Table III 同时说明：CLNet 的**编码器反而略重于** CRNet，解码器更轻，存储占用大致相同——
总 FLOPs 的下降主要来自 BS 侧解码器，UE 侧并未变轻。

CVLNet 进一步把复值组件放进多尺度特征增强编码器和多分辨率 X 形重建解码器，摘要称其编码器
参数量与当时的 SOTA 轻量网络处于同一水平，而在高压缩比下精度最多提升 33.4% [4]。Liu et al.
则构造了复值编码器—解码器，用复卷积下采样（CCD）提取特征、用复密集块（CDBlock）重建，
摘要报告相对若干经典深度学习方法平均精度提升 17.5% [5]。这两篇的摘要都未给出数据集、
epoch 预算与“精度”的具体定义，相关数值只能视为作者自报的方向性结论。

ENet 处于复值与轻量化的交界：它仍分别处理实部与虚部，但利用角时域复响应的相关性，使为实部
训练的网络可以直接复用于虚部，从而把网络规模相对已有方案缩小近一个数量级 [6]。这说明“利用
实虚相关”并不必然要求复数算术。

### 3.3 复值线索的观察

综合来看，CSI 反馈中的“复值”多数停留在输入嵌入或少量复值组件层面，完整复值反向传播的
工作较少；CLNet 对复数核优化困难与开销高的判断，是这一现象的公开理由之一 [3]。另一方面，
复值工作的收益多以“精度提升百分比”表述，而非 NMSE(dB)，且数据集与训练预算披露不齐，
难以与固定 CR 的 NMSE 表直接对齐。本仓库的 `csifeedback/models/clnet.py` 保留了 1×1+SE
分支、空间注意力与 hard-sigmoid，可作为复现原文结构的参照。

## 4 轻量化：结构、训练与离散化

### 4.1 轻量结构

ShuffleCsiNet 在 ConvCsiNet 的基础上，用深度可分离卷积与通道混洗替换编码器的常规卷积下采样，
解码器保持不变 [7]。其 COST2100 设定为 32 天线、`Nc=256` 子载波、75,000/12,500/12,500 划分、
batch 200、1000 epoch、CR 1/16 与 1/32；CR=1/32 时参数量降至 ConvCsiNet 的 21.05%，FLOPs
亦明显下降。这里有两个协议细节值得注意：数据划分与 CsiNet 的 100,000/30,000/20,000 不同；
轻量化只发生在 UE 侧编码器。

STNet 用局部组自注意力与全局下采样注意力构成空间可分离注意力，替代完整多头注意力；摘要称
其在部分场景以约 1/10 的资源量达到 SOTA 水平 [8]，但未在摘要中拆分参数量与 FLOPs。STNet 是
本仓库的三个 baseline 之一，其具体训练协议以 `docs/baselines.md` 为准。

2026 年的 SSE-Net 把行、列位置信息编码进 CSI 预处理，并组合多尺度可分离卷积与自适应交叉
注意力；摘要报告该预处理使训练收敛速度提升 46%，同时降低反馈开销 [23]。

同年 SpikingCSINet 以脉冲实现反馈与主干计算，并用渐进残差结构沿时间步编码残差，以缓解
二值脉冲在高维重建中的信息瓶颈 [22]。作者在 COST2100 上称其性能—效率折中优于轻量卷积
基线，且能耗较 Transformer 类方法降低 93% 以上；摘要未给出参数量、时延与能耗测量平台。

### 4.2 知识蒸馏

Tang et al. 最早把教师—学生蒸馏用于 CSI 反馈的轻量化 [9]；该会议论文给出早期验证。Cui et al.
的期刊版本随后补充了 encoder KD 与保护知识产权的变体，完整方案为：教师是 CRNet，
学生是编码器简化的 CRNet-SE；提出 autoencoder KD（学生模仿教师的重建输出）、encoder KD
（只蒸馏编码器、直接复用教师解码器以省训练开销）以及不传输编码器结构的变体 [10]。

其数据由 QuaDRiGa 生成（UMi-LOS、2.655 GHz、70 MHz 带宽、32 子载波、32 天线，
100,000/30,000/20,000 划分），batch 200，验证集 MSE 连续 50 epoch 不下降即停止。原文报告
学生在 CR=1/16 与 1/32 时的 FLOPs 分别为教师的 5.28% 与 3.05%，推理时间为教师的
13.80%–14.76%；同时明确指出学生性能不会超过教师 [10]。由于数据来自 QuaDRiGa 而非
COST2100，这组比例不能直接搬到本仓库的 baseline 表上。

### 4.3 量化、二值化与熵编码

JC-ResNet 用联合卷积残差网络同时做信道特征提取与恢复，并把设计目标放在比特级量化性能上，
而不只是最小化连续值重建误差 [11]；摘要称其显著优于对照方法。
BCsiNet 只对 UE 侧编码器做二值化，摘要报告相对 CsiNet 超过 30 倍的内存节省、约 2 倍推理
加速，且反馈性能可与原 CsiNet 相比 [12]。ACRNet 则把网络聚合、PReLU、“弹性反馈”（按资源
限制调整网络）、二值化与特征量化组合在一起 [13]；该文目前可核实的形式是 2021 年的 arXiv
预印本，正式发表信息未核实。

Shape-Gain VQ 在 VQ-VAE 框架下做有限码率反馈：潜向量幅度用非均匀标量码本量化，方向用可训练
的 Grassmannian 码本量化，并以嵌套码本支持多码率 [14]。摘要称其在降低 VQ 计算复杂度的同时
改善了给定开销下的重建；作者还给出嵌套码本的多码率选择规则，使单一码本服务不同反馈开销。

NTC 把 CSI 视为高维复值“图像”，用 Swin Transformer 学习非线性变换与潜变量分布，再以
嵌套格点均匀量化与熵编码生成比特流 [15]。其在 COST2100 indoor 5.3 GHz、`(Nt, N~c, Nc) =
(32, 1024, 32)` 上报告：嵌入维度取 24 时模型仅约 130,000 参数，约为对照 SOTA 的 6%，
且单模型 200 epoch 即可，而对照 CH-CsiNet-Pro 需要 5 个模型、每个 2,000 epoch；作者也指出
NTC 类方法在高码率区出现 NMSE 饱和。

STQENet 沿用 STNet 的可分离注意力与 CRBlock 结构，加入 μ-law 量化、直通估计器与 Huffman
熵编码，做端到端训练 [16]。
其 COST2100 设定为 32 天线、32 子载波、100,000/30,000/20,000 划分、batch 200、1000 epoch、
Adam 1e-3，比较 2/4/6/8 bit 下的 BPP–NMSE。摘要报告相对多个对照平均改进 21.5%（indoor）与
18.9%（outdoor），但未说明该百分比是相对 NMSE 降幅还是其他口径。

### 4.4 剪枝与通用压缩

Guo et al. 的综述性文章系统讨论了通信网络中神经网络的压缩与加速，把剪枝、量化、低秩分解、
知识蒸馏等列为代表性手段，并给出两个案例研究 [19]。在 CSI 反馈上，Erak & Abou-Zeid
对 CsiNet 做了一项较完整的“实测型”压缩研究：50% 幅度剪枝、训练后动态范围量化（8 bit）、
每层 32 簇的权重聚类，并部署到 Raspberry Pi 4 上用 TensorFlow Lite 测时延 [17]。其报告为：
剪枝平均减体积 43.9%、减时延 66.8%（依赖 XNNPack 稀疏算子）；量化减体积 73.1%、减时延
10.4%；权重聚类减体积 79.2% 但几乎不降时延；剪枝+量化减体积 86.5%、减时延 76.2%。作者
特别指出，若没有稀疏推理库，剪枝带来的时延收益基本消失；outdoor、CR=1/16 时剪枝模型的
NMSE 平均比原模型差 15.3%。

SNIP-CsiNet 用第一个 epoch 的梯度信息做单次剪枝，省去预训练与复杂剪枝日程；摘要称在相同
压缩率下重建效果与对照相当甚至更好，并更有效地降低复杂度 [18]。该文未在摘要中给出参数量
或 FLOPs 的具体数值。

### 4.5 低秩

“低秩”在 CSI 反馈文献中有两种不同含义，需要区分。其一是**网络权重的低秩分解**：Guo et al.
把它列为通用压缩手段 [19]，但本次检索未找到专门对 CSI 反馈自编码器权重做低秩分解、并给出
完整参数量/FLOPs 报告的论文，故此处不作为独立方法展开。

其二是**信道本身的低秩先验**。TSLR 面向 mmWave massive MIMO，采用两级低秩结构并展开 FISTA
得到 FISTA-Net，配合基于注意力的自适应阈值网络（原文称 ST-Net，与本仓库 baseline STNet
无关）；摘要称其在多种场景优于既有算法 [20]。
LRP 则在 UE 侧做可学习的秩一感知、在 BS 侧做秩一合成，并用 DCRNetV2 的门控空洞残差路径
补偿有限秩误差；摘要称其以低于迭代模型驱动方法的复杂度取得更好的精度—复杂度折中 [21]。
两篇都属于模型驱动/结构化压缩，与第 4.1–4.4 节的“网络压缩”不是同一类手段。

## 5 复杂度报告口径

轻量化论文报告的“复杂度”至少有六种量：参数量、FLOPs/MACs、模型文件体积、推理时延、
峰值内存/能耗，以及反馈侧的 BPP。它们的统计边界差异很大：

- **算子边界**：是否计入 BN、激活、池化与全连接。ShuffleCsiNet 明确给出卷积、深度卷积、
  BN 与全连接的参数量公式，并把卷积、深度卷积、池化、全连接计入 FLOPs [7]；Cui et al. 的
  KD 论文则明确忽略 BN 与激活，只统计卷积与全连接 [10]。
- **位置边界**：是否把编码器与解码器分开。CLNet 的总 FLOPs 下降但编码器略重 [3]；
  ShuffleCsiNet 只改编码器 [7]。对“UE 侧是否可部署”这一问题，总 FLOPs 并不是充分的量。
- **实现边界**：理论稀疏/理论 FLOPs 与实测时延的差距。Erak & Abou-Zeid 的结果表明，
  同一剪枝模型在没有稀疏加速库时几乎不提速 [17]。
- **训练边界**：epoch 数、模型个数、是否需要教师。NTC 以“单模型 200 epoch”对比对照的
  “5 个模型 × 2,000 epoch” [15]；KD 的收益还依赖一个已训练好的教师 [10]。
- **指标定义**：CLNet 原文把 FLOPs 写成 “floating-point operations per second”，实际统计的
  是计算量；多篇论文的摘要以“精度提升百分比”表述收益，而非 NMSE(dB)，且未说明基准
  [3][4][5][16]。
- **数据协议**：同为 COST2100，划分可能是 100,000/30,000/20,000，也可能是
  75,000/12,500/12,500；NTC 甚至以 BPP–NMSE 曲线而非固定 CR 报告 [15]。

| 工作 | 报告的量 | 原文数值 | 训练/测试设定 | 协议备注 |
|---|---|---|---|---|
| CLNet [3] | FLOPs、编解码器分解、存储 | 相对 CRNet −24.1% FLOPs、+5.41% 精度 | COST2100 100k/30k/20k；1000 epoch；2080 Ti | 编码器反而略重；精度未换算为 dB |
| ShuffleCsiNet [7] | 参数量、FLOPs | CR=1/32 时参数为 ConvCsiNet 的 21.05% | COST2100；75k/12.5k/12.5k；1000 epoch | 划分与 CsiNet 不同；仅编码器轻量 |
| Cui et al. KD [10] | FLOPs、推理时间 | 学生为教师 3.05%–5.28% FLOPs、13.80%–14.76% 时延 | QuaDRiGa UMi-LOS；32 子载波；batch 200 | 非 COST2100 |
| Erak & Abou-Zeid [17] | 模型体积、实测时延 | −86.5% 体积、−76.2% 时延（剪枝+量化） | CsiNet；Raspberry Pi 4；TFLite/XNNPack | 依赖稀疏加速库；outdoor 有精度损失 |
| Park et al. NTC [15] | 参数量、训练成本 | 约 130k 参数（约为对照的 6%）；单模型 200 epoch | COST2100 indoor；32×1024×32 | 以 BPP–NMSE 报告，非固定 CR |
| STNet [8] | “资源量” | 约 1/10 | COST2100 | 未拆分参数量与 FLOPs |

## 6 小结

就文献事实而言：（1）复值设计在 2021–2022 年集中出现，但多数以输入嵌入或少量复值组件实现，
完整复值网络较少，CLNet 把原因归结为复数核的优化困难与开销 [3]；（2）轻量化沿结构、训练
策略、离散化三条线并行推进，2024 年后量化与熵编码逐渐与自编码器端到端联合训练 [14][15][16]；
（3）复杂度报告高度异质，算子边界、位置边界与实现边界都不统一，跨论文的“更轻”结论需要
回到原始表格与协议；（4）UE 侧与 BS 侧的开销常不对称，总 FLOPs 下降不等于 UE 侧下降 [3][7]；
（5）2025–2026 年的新工作更强调参数效率、码率—失真折中与新计算范式（如脉冲网络），而不是
单一固定 CR 下的 NMSE 排名 [15][22][23]。

## 7 文献明确指出的开放问题

- **复数核的优化**：CLNet 指出复数核难以反向传播训练、复杂度高，因而转向伪复值输入 [3]。
- **稀疏收益的落地**：Erak & Abou-Zeid 指出没有稀疏推理库时剪枝几乎不降时延，且 outdoor
  低压缩比下剪枝会带来精度损失 [17]。
- **蒸馏的上界与泛化**：Cui et al. 指出学生性能不能超过教师，并用跨场景、跨环境、跨带宽的
  实验考察变体方法的泛化能力 [10]。
- **熵模型的表达力**：NTC 采用独立的因子化先验，作者把更复杂的分布建模留作未来工作，并
  报告高码率区 NMSE 饱和 [15]。
- **粗量化的失真地板**：STQENet 观察到 2 bit 量化在高 SNR 下出现性能地板，粗量化成为主要
  失真源 [16]。
- **模型驱动方法的时延**：LRP 把迭代模型驱动方法的高复杂度与高时延作为自身动机 [21]。

## 参考文献

[1] Trabelsi C. et al. Deep Complex Networks. ICLR, 2018. https://arxiv.org/abs/1705.09792

[2] Zhang Y. et al. CV-3DCNN: Complex-Valued Deep Learning for CSI Prediction in FDD Massive MIMO Systems. IEEE Wireless Communications Letters, 10(2):266–270, 2021. https://doi.org/10.1109/LWC.2020.3027774

[3] Ji S., Li M. CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 10(10):2318–2322, 2021. https://arxiv.org/abs/2102.07507

[4] Li H. et al. CVLNet: A Complex-Valued Lightweight Network for CSI Feedback. IEEE Wireless Communications Letters, 11(5):1092–1096, 2022. https://doi.org/10.1109/LWC.2022.3157263

[5] Liu Q. et al. CSI Feedback Based on Complex Neural Network for Massive MIMO Systems. IEEE Access, 10:78414–78422, 2022. https://doi.org/10.1109/ACCESS.2022.3194035

[6] Sun Y. et al. A Lightweight Deep Network for Efficient CSI Feedback in Massive MIMO Systems. IEEE Wireless Communications Letters, 10(8):1840–1844, 2021. https://doi.org/10.1109/LWC.2021.3083331

[7] Cao Z. et al. Lightweight Convolutional Neural Networks for CSI Feedback in Massive MIMO. IEEE Communications Letters, 25(8):2624–2628, 2021. https://arxiv.org/abs/2005.00438

[8] Mourya S. et al. A Spatially Separable Attention Mechanism for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 12(1):40–44, 2023. https://arxiv.org/abs/2208.03369

[9] Tang H. et al. Knowledge-Distillation-Aided Lightweight Neural Network for Massive MIMO CSI Feedback. IEEE VTC2021-Fall, pp. 1–5, 2021. https://doi.org/10.1109/VTC2021-Fall52928.2021.9625047

[10] Cui Y. et al. Lightweight Neural Network With Knowledge Distillation for CSI Feedback. IEEE Transactions on Communications, 72(8):4917–4929, 2024. https://arxiv.org/abs/2210.17113

[11] Lu C. et al. Bit-Level Optimized Neural Network for Multi-Antenna Channel Quantization. IEEE Wireless Communications Letters, 9(1):87–90, 2020. https://doi.org/10.1109/LWC.2019.2942908

[12] Lu Z. et al. Binary Neural Network Aided CSI Feedback in Massive MIMO System. IEEE Wireless Communications Letters, 10(6):1305–1308, 2021. https://doi.org/10.1109/LWC.2021.3064963

[13] Lu Z. et al. Binarized Aggregated Network with Quantization: Flexible Deep Learning Deployment for CSI Feedback in Massive MIMO System. arXiv:2105.00354, 2021. https://arxiv.org/abs/2105.00354

[14] Shin J. et al. Vector Quantization for Deep-Learning-Based CSI Feedback in Massive MIMO Systems. IEEE Wireless Communications Letters, 13(9):2382–2386, 2024. https://arxiv.org/abs/2403.07355

[15] Park B. et al. Transformer-Based Nonlinear Transform Coding for Multi-Rate CSI Compression in MIMO-OFDM Systems. arXiv:2502.19847, 2025. https://arxiv.org/abs/2502.19847

[16] Ansarifard M. et al. CSI Compression Beyond Latents: End-to-End Hybrid Attention-CNN Networks with Entropy Regularization. arXiv:2509.08776, 2025. https://arxiv.org/abs/2509.08776

[17] Erak O., Abou-Zeid H. Accelerating and Compressing Deep Neural Networks for Massive MIMO CSI Feedback. IEEE ICC, pp. 1029–1035, 2023. https://arxiv.org/abs/2304.01914

[18] Cui Y. et al. Lightweight Neural Network Based SNIP for CSI Feedback in Massive MIMO. Electronics Letters, 59(19), 2023. https://doi.org/10.1049/ell2.12843

[19] Guo J. et al. Compression and Acceleration of Neural Networks for Communications. IEEE Wireless Communications, 27(4):110–117, 2020. https://doi.org/10.1109/MWC.001.1900473

[20] Guo J. et al. CSI Feedback With Model-Driven Deep Learning of Massive MIMO Systems. IEEE Communications Letters, 26(3):547–551, 2022. https://arxiv.org/abs/2112.06405

[21] Tang S. et al. Low-Rank Prior-Guided Rank-One Sensing for Efficient CSI Feedback. arXiv:2609.28156, 2026. https://arxiv.org/abs/2609.28156

[22] Liu Y. et al. Massive MIMO CSI Feedback With Spiking Neural Networks. IEEE Wireless Communications Letters, 15:4862–4866, 2026. https://arxiv.org/abs/2605.12434

[23] Liu M. et al. Structured Spatial Encoding Network for Efficient CSI Feedback in FDD Massive MIMO Systems. IEEE Wireless Communications Letters, 15:2219–2223, 2026. https://doi.org/10.1109/LWC.2026.3675355

## 核查记录

**来源层级**

- 原文全文（arXiv HTML）：CLNet [3]（v3）、Cui et al. KD [10]（v3）、Erak & Abou-Zeid [17]
  （v1）、ConvCsiNet/ShuffleCsiNet [7]（v1）、Park et al. NTC [15]（v1）、STQENet [16]（v1）。
  这些条目的方法细节、训练设定与数值来自全文。
- 仅摘要：CV-3DCNN [2]、CVLNet [4]、Liu et al. [5]、ENet [6]、JC-ResNet [11]、BCsiNet [12]、
  ACRNet [13]、Shape-Gain VQ [14]、SNIP-CsiNet [18]、TSLR [20]、LRP [21]、SpikingCSINet [22]、
  SSE-Net [23]、STNet [8]、Tang et al. [9]、Guo et al. 综述 [19]、
  Deep Complex Networks [1]。其中数据集、epoch 或“精度”定义未在摘要出现者，正文已注明
  “摘要未给出”或未引用其数值。
- 二手来源：未使用综述或博客中的数值。Erak & Abou-Zeid 对 ShuffleCsiNet“约 5 倍参数下降”
  的转述未被采用，正文改用原文的“CR=1/32 时为 ConvCsiNet 的 21.05%”。

**元数据核实**

- 题名、作者、年份、卷期页码与 DOI 通过 OpenAlex 与 DOI/arXiv 题名页交叉核对。CLNet 的
  正确 venue 是 IEEE Wireless Communications Letters（10(10):2318–2322），不是部分二手
  文献所写的 IEEE Communications Letters。
- STQENet 的 arXiv 现题名为 “CSI Compression Beyond Latents: End-to-End Hybrid Attention-CNN
  Networks with Entropy Regularization”，正文中的 STQENet 是该文提出的网络名；本文按现题名
  著录。
- SNIP-CsiNet 的卷期为 Electronics Letters 59(19), 2023（部分引用写作 2024）。

**数值口径的不确定性**

- CLNet 原文 Table I 的绝对 NMSE 数值未被抽取，本文只引用其相对 CRNet 的百分比与 FLOPs
  百分比；原文的“精度提升”未说明是否等价于相对 NMSE 降幅。
- CVLNet、Liu et al.、STQENet 的“精度提升百分比”均未给出可换算为 NMSE 的定义，本文按
  “作者自报”转述，不作为可比较的 NMSE 结论。
- CVLNet、Liu et al.、JC-ResNet、SNIP-CsiNet、TSLR、LRP、SpikingCSINet 的数据集与训练预算
  未在摘要中完整披露，正文未为其附加 COST2100 或 CR 标签。
- STNet 的“约 1/10 资源量”未拆分参数量与 FLOPs，本文原样保留该表述。

**检索词与日期**

检索日期 2026-11-03。检索词：`complex-valued CSI feedback`、`complex convolution CSI feedback`、
`CLNet complex lightweight CSI feedback`、`lightweight CSI feedback`、`CSI feedback quantization`、
`binary neural network CSI feedback`、`CSI feedback pruning`、`knowledge distillation CSI feedback`、
`low-rank CSI feedback`、`efficient CSI feedback`、`spiking neural network CSI feedback`。
筛选标准：2018 年后、与下行 CSI 压缩反馈直接相关、可核实元数据；共检视约 30 条候选，
保留 23 条。
