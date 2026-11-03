# 模型驱动、压缩感知与物理先验

> 状态：调研笔记（检索与核对日期 2026-11-03）。本文件只记录文献事实与协议差异，
> 不提出新方法，也不对跨协议结果做排名式判断。

## 1. 范围与检索方法

本文件收录把「信道为何可压缩」的模型知识写进反馈链路的做法，包括：

- 压缩感知（CS）的稀疏、联合稀疏与低秩假设；
- 角度-时延域的截断、变换基与字典；
- 多径参数化信道模型（幅度、相位、到达角、时延）；
- 二阶统计量先验（角度-时延功率图）；
- 把迭代重建算法展开成网络（unrolled networks，ISTA/FISTA/AMP）；
- 把网络本身当作先验（未训练网络、plug-and-play 去噪先验）。

检索在 arXiv API 与 arXiv HTML 全文上完成，检索词包括 `"CSI feedback" AND "compressed
sensing"`、`"CSI feedback" AND "deep unfolding"`、`"CSI feedback" AND "low-rank"`、
`"CSI feedback" AND prior`、`"CSI feedback" AND "model-based"`、`"untrained" AND CSI`、
`"plug-and-play" AND CSI`、`"approximate message passing" AND "CSI feedback"`，并按提交
日期倒序复查 2025-2026 年条目；venue 与卷期另用 WebSearch 与出版社页交叉核对。范围外：
纯量化/熵编码、纯时序预测、纯架构搜索（见 03、06 册）。

## 2. 时间线与分类

**2012-2018：CS 是主线。** Rao 与 Lau 把多用户 CSIT 的联合稀疏性用于分布式压缩与反馈 [1]：
「稀疏基 + 随机测量矩阵 + ℓ1/贪心恢复」，反馈量以测量数计，重建在基站侧迭代完成。

**2018-2020：CS 从算法变成网络的设计约束。** CsiNet 沿用角度-时延域截断作为固定预处理，
把 32×32 的截断矩阵交给卷积自编码器 [5]；CsiNet+ 以「多速率压缩感知」为框架，报告
SM-CsiNet+ 与 PM-CsiNet+ 分别把 UE 侧参数量减少 38.0% 与 46.7% [6]。同期出现两种混合：
Wu 等只学习测量矩阵，重建仍交给线性规划求解器 [3]；Groß 等把近似消息传递（AMP）展开成
权值共享的网络，求解 5G NR 显式 CSI 的行稀疏多测量向量问题，并说明可在纯合成数据上
预训练 [4]。

**2021-2022：展开网络成为独立分支。** FISTA-Net 展开 FISTA，用基于注意力的 ST-Net 自适应
选择阈值，并配合两阶段低秩（TSLR）方案 [10]；LORA 展开 ISTA，用可学习正则模块替代 ℓ1 范数，
并与量化器联合训练 [11]；SPTM2-ISTANet+ 在 ISTA 展开中加入球面归一化与可训练测量矩阵 [12]。

**2023-2026：从「展开迭代」转向「把结构写进编解码器」。** L2O 方案在 UE 侧只用一个线性
可学习投影，在 BS 侧用可学习稀疏变换加逐元素 LSTM 重建 [13]；LRP 直接在秩一分量上做
可学习感知与合成，绕开迭代恢复 [14]；HS-PINNnet 让网络输出多径参数，再由信道模型重建
CSI [15]；曲线波分解 [16]、角度-时延功率图先验 [17]、Loewner 插值动态基 [18]、生成式
先验 [19] 同属此类。本次检索中，纯 ISTA/FISTA 展开工作集中在 2021-2022 年，2024 年后
多为「结构化先验 + 端到端编解码」。

**表 1　按先验形式分类（同一工作可能跨类）**

| 类别 | 核心先验 | 代表工作 | UE 侧 | BS 侧 |
|------|----------|----------|-------|-------|
| 传统 CS | 角度域稀疏、联合稀疏 | [1] | 随机/固定测量矩阵 | ℓ1 或贪心迭代恢复 |
| 可学习测量与字典 | 稀疏基可学习 | [2][3][4] | 学习到的测量矩阵或字典 | 线性规划 / 展开 AMP |
| 展开网络 | 迭代算法的算子结构 | [10][11][12] | 测量矩阵或浅层编码器 | 固定层数的迭代展开 |
| 参数化物理先验 | 有限多径叠加 | [14][15] | 感知/编码多径特征 | 参数估计 + 信道模型重建 |
| 统计与变换域先验 | 二阶统计、变换域稀疏 | [16][17][18] | 变换或 token 选择 | 反变换 / 先验条件化解码 |
| 未训练与 PnP 先验 | 网络结构或去噪器 | [20][21][22][23] | 固定线性投影 | 迭代优化中插入去噪器 |

## 3. 代表性工作

### 3.1 压缩感知与可学习的测量/字典

Rao 与 Lau 的工作 [1] 是这一支线的参照点：它把多用户信道在角度域的联合稀疏作为先验，
反馈的是压缩测量而非原始系数；其结论依赖稀疏度与测量矩阵性质，属于「先验成立则有效」
的典型表述。

Gadamsetty 等 [2] 把字典从固定的 DFT 基换成学习到的公共字典，给出 K-SVD 与正交 Procrustes
两种构造，覆盖单用户与多用户。作者报告其估计信道向量的 NMSE 低于固定 DFT 字典，并把反馈量
降低 50%、UE 与 BS 侧存储降低至少 50%。这是少数不依赖深度网络、但把「基的选择」交给数据的
工作。

Wu 等 [3] 的做法介于两者之间：用 ℓ1-minimization autoencoder 只优化测量矩阵，重建仍由传统
稀疏求解器完成，因此网络规模很小、可解释性强，但推理时仍有迭代求解开销。

Groß 等 [4] 面向 5G NR 的显式 CSI 反馈，比较了多种低复杂度 CS 算法，并提出基于学习 AMP 的
行稀疏 MMV 网络。作者强调其权值共享带来低内存与快速前向，且可在纯合成数据上预训练——这两点
与「必须采集大量真实 CSI 配对」的纯数据驱动路线形成直接对照。该文发表于 International ITG
Workshop on Smart Antennas (WSA) 2021。

### 3.2 展开网络

展开网络把「迭代算法的每一次更新」映射成一层网络，从而在保留算子结构的同时学习步长、阈值
或正则项。方法源头是 LISTA [7]；面向通信系统的综述见 [8][25]，面向物理层的模型驱动综述
见 [9]。

FISTA-Net [10] 展开 FISTA，并把收缩阈值交给一个带注意力机制的 ST-Net 自适应选择；配合
两阶段低秩方案处理毫米波信道的低秩结构。LORA [11] 的关键改动是把 ℓ1 正则替换为可学习的
正则模块，使正则项不再有显式形式，同时给出可学习量化器与联合训练方案。

LORA 全文给出两点与协议直接相关的事实：其一，展开类方法的编码器通常只是一个测量矩阵，
参数量集中在解码器，作者明确认为这在「UE 受限、BS 资源充足」的部署假设下是合理的；其二，
量化器与展开网络联合训练时 NMSE 为 0.0053，先训练网络再训练量化器时为 0.0069（CR=1/4，
3GPP TR 38.901 RMANLOS 场景，1000 epoch、batch 200、学习率 1e-3）。这说明量化是否在环
会改变结论，而很多论文报告的是无量化或理想反馈下的 NMSE。

SPTM2-ISTANet+ [12] 在 ISTA 展开中加入球面归一化以处理路损差异，并用可训练测量矩阵与残差
恢复块提升小样本、跨场景时的表现；作者进一步用该网络作为锚点模型，配合轻量翻译模块适配新
环境。其贡献重心是「复用与适配」，而不是单一场景的精度。

### 3.3 低秩与多径参数化

低秩先验在毫米波与宽带上比「逐元素稀疏」更贴近物理：信道近似为若干秩一分量之和。TSLR [10]
用两阶段低秩处理这一结构；LRP [14] 则把低秩写进网络本身——UE 侧做可学习的秩一感知，BS 侧
做秩一合成，从而不需要迭代恢复。LRP 在 COST2100 indoor、CR=1/4 下报告 −13.18 dB NMSE、
0.68 M FLOPs，同表中最强的模型驱动对照 FISTA 为 −10.46 dB、42.99 M FLOPs（训练条件见第 5
节）。作者说明有限秩近似会留下残差，需用门控空洞卷积残差分支补偿（DCRNetV2）。

HS-PINNnet [15] 走得更彻底：网络不直接输出 CSI，而是输出每条径的幅度、相位、到达角与时延，
再由多径信道模型合成 CSI；训练分两步，先最小化参数误差，再最小化重建 NMSE。作者给出误差
传播的一阶分析：幅度与相位误差不随天线/子载波索引累积，而时延误差乘以子载波索引、角度误差
乘以天线索引，因此在 32 天线、1% 角度误差下可造成接近 50% 的失真。这一分析解释了为什么
参数化方法对角度估计精度格外敏感。作者报告相对 SOTA 减少 92.8% FLOPs、FPGA 仿真延迟低两个
数量级；该文注脚说明较早版本被 IEEE ICC 2026 接收。

### 3.4 统计与变换域先验

PM-VQ [17] 把「平均角度-时延功率图」作为收发两端共享的二阶统计先验，编码器输出 token，
用空间自适应掩码挑选在预算内的 token，再向量量化并回传位置；解码器在 BS 侧先解掩码、再用
先验条件化解码。作者在 Sionna 生成的 3GPP TR 38.901 UMa 数据上评测，重点是能量弥散在大量
角度-时延系数上的 diffuse 区域，并报告在该区域所有测试 SNR 与反馈维度上 NMSE 最低；作者
还指出，即使用少量信道样本估计该功率图，先验仍然有益。

SwinCANet [16] 用曲线波变换把 CSI 分解为低频与高频分量，分别用 Swin Transformer 与通道
注意力处理，再做多尺度融合；变体 De-SwinCANet 用 Sigmoid 阈值抑制噪声系数。Loewner 插值
方案 [18] 针对 U6G（6425-7125 MHz）宽带场景指出：DFT 基的能量泄漏会随带宽增大而加重，
因此改用由当前 CSI 生成的动态基（Loewner 插值）在频域压缩，再用网络压缩空间域，并用
tail dropout 与多目标训练得到可变长度码字。这两项工作的共同点是「换掉固定的 DFT 基」，
但都以 2026 年 arXiv 预印本形式出现，尚未见到同行评议版本。

### 3.5 未训练、无监督与 plug-and-play

Deep Image Prior [20] 的前提是：随机初始化网络的结构本身就是可用先验，无需外部训练数据。
搬到 CSI 的工作不多，且多不在标准 CR 网格上：Vilas Boas 等 [21] 用未训练网络拟合少量测量
来「学习传播环境」，再迁移给邻近用户，并指出欠参数化有助于低开销上报；其定位是重建/上报，
而非固定压缩比的反馈压缩。

更常见的是「有训练但无配对标签」的 PnP 路线。Wan 等 [22] 无监督训练一个去噪器，作为 PnP
先验插入交替优化，用同一模型完成信道估计、天线外推与 CSI 反馈，在 DeepMIMO 上无需按任务
微调。CSI-PPPNet [23] 更贴近部署：UE 只做线性投影，BS 在交替优化中用 DL 去噪器替代先验的
近端算子，一个去噪模型可复用于任意压缩比；作者把「减少模型存储、免除联合训练与模型下发」
列为主要动机。

需要如实说明：本次检索未找到把逐样本 DIP 直接用于 COST2100 标准 CR 网格并给出 NMSE/rho 的
工作；PnP 与未训练网络更多出现在信道估计、重建与上报任务中。

### 3.6 协议层的混合：生成式先验

CsiCoGen [19] 把有限比特的决策放在反向扩散轨迹上的「码本约束高斯创新选择」：前缀长度 L
对应 R_L = L log2 K 比特，码本离线生成、不随信令传输，去噪器作为共享 CSI 先验预训练。
作者在 COST2100 上给出 792 与 1592 比特两组 NMSE/rho（数值见第 5 节），并注明已被 IEEE
Wireless Communications Letters 接收。这类方法以比特预算定义速率，不能与本仓库 CR=1/4
（512 维浮点）直接对齐，详见第 4 节；生成式路线的系统梳理见 05 册。

## 4. 与纯数据驱动自编码器的协议差异

这一节是本文件的核心。把模型驱动方法与 CsiNet/CLNet/STNet 这类自编码器放在一张表里比较
之前，至少要先对齐以下五项。

**4.1 反馈对象与速率。** 纯数据驱动方法反馈 M 维浮点向量，CR = M/(2N_tN_c)（LORA 采用
此定义 [11]）。模型驱动方法有两种偏离：继续用浮点维度但在解码端附加迭代/展开 [10][12][14]；
或直接反馈比特，速率由比特预算或可变长度码字定义 [17][18][19]。二者不是同一个量；量化是否
在环也会改变 NMSE（LORA 联合训练 0.0053、分步训练 0.0069）。因此「在 CR=1/16 更好」必须
同时说明浮点还是比特、量化位数与是否含熵编码。

**4.2 监督。** 纯自编码器只需 CSI 配对；模型驱动方法还可能需要多径参数标签 [15]，或允许
合成数据预训练 [4]、无监督/单侧训练 [22][23]、测试时拟合 [21]。监督不同，样本量与标注成本
不同，「同等训练预算」这一前提未必成立。

**4.3 计算位置。** UE 侧只保留测量矩阵或线性投影、BS 侧承担迭代与参数估计，是常见分工
[11][14][23]，LORA 明确以此作为设计理由。这与 CLNet/STNet 两端都放卷积网络的做法不冲突，
但「总参数量」会掩盖两端分布差异。

**4.4 数据与信道模型。** COST2100 常见用法是 32 天线、1024 子载波、截断到 32 行，得到
32×32 [14]，与本仓库 (2,32,32) 维度一致，但截断、归一化与划分未必相同；另有 3GPP TR
38.901 [11]、Sionna [14][17]、DeepMIMO [22] 与实测数据 [14]。跨信道模型的结论可迁移性
本身未被验证。

**4.5 指标与选择。** 本仓库同时报告 NMSE 与 rho；所核查的模型驱动论文大多只报 NMSE，
rho 仅部分出现 [19]；归一化方式与是否含量化误差也并非每篇都写清。此外，所核查论文很少
写明 checkpoint 选择策略，「训练最优值」与「固定 checkpoint 复评值」通常无法区分。

**表 2　四类方法的协议对照（不含性能排名）**

| 维度 | 纯数据驱动自编码器 | 展开网络 | 参数化物理先验 | 未训练 / PnP |
|------|--------------------|----------|----------------|--------------|
| 反馈表示 | M 维浮点（可接量化） | M 维浮点或测量向量 | 多径特征 / 秩一系数 | 线性投影测量 |
| 训练监督 | CSI 配对 | CSI 配对（可加正则学习） | 参数标签 + CSI（两阶段） | 无配对标签 / 单侧 / 测试时拟合 |
| 推理计算 | 一次性前向 | 固定层数展开 | 参数估计 + 模型合成 | BS 侧迭代 + 去噪器 |
| 速率灵活性 | 每个 CR 一个模型或可变速率设计 | 通常每 CR 训练 | 取决于码字维度 | 单模型可跨 CR（PnP） |
| 代表工作 | [5][6] | [10][11][12] | [14][15] | [21][22][23] |

## 5. 可核实的数值示例

只列两个原文读到、条件完整的例子，不作横向比较。

- LRP [14]：COST2100 indoor，CR=1/4，编码秩 128、解码秩 4，1500 epoch、batch 200、
  初始学习率 2e-3，训练/验证/测试 100k/30k/20k。报告 −13.18 dB NMSE、0.68 M FLOPs；
  同表的 FISTA 为 −10.46 dB、42.99 M FLOPs；全文未见 checkpoint 选择方式的说明。
- CsiCoGen [19]：COST2100 indoor/outdoor，792 比特时 −28.58/−13.96 dB、rho 0.9964/0.9597；
  1592 比特时 −30.72/−20.37 dB、rho 0.9967/0.9748。其速率以比特预算计，与本仓库的浮点
  CR 定义不同，不能与浮点 CR=1/4 的自编码器结果直接对应。

其余工作（[4][10][11][12][15][16][17][18]）的摘要只给出相对结论或复杂度比例，未见可直接
摘录的绝对 NMSE/rho；本文件因此不填写这些数值。

## 6. 小结（仅陈述文献事实）

- 先验形式从「角度域稀疏」扩展到「低秩」「有限多径参数」「二阶统计」「生成式先验」，
  网络位置亦从解码端迭代前移到编码器与码字结构。
- 展开网络在 2021-2022 年集中出现，其共同动机是保留迭代算法的算子结构，并把阈值、正则或
  步长交给学习 [10][11][12]；2024 年之后的工作更多把结构直接嵌入编解码器 [14][15][17]。
- 反馈量的定义并不统一：浮点维度、比特预算、可变长度码字三种口径并存 [11][17][19]，
  量化是否在环会改变 NMSE 量级 [11]。
- 未训练网络与 PnP 在 CSI 反馈上的直接证据少于其在信道估计/重建上的证据 [21][22][23]；
  逐样本 DIP 式的标准 CR 网格结果，本次未检索到。

## 7. 文献明确指出的开放问题

- 模型驱动 DL 依赖领域知识建模，泛化与模型失配被综述列为待研究事项 [9]；展开网络的
  开放问题清单见 [8]（摘要只说明结尾列出清单，具体条目需查原文）。
- LORA 明确提出量化与展开网络的联合训练、以及 UE/BS 参数分布不对称带来的部署问题 [11]。
- Learned AMP 的复杂度优势与真实信道失配下的鲁棒性需分别评估 [4]；单侧模型跨 CR 复用是
  CSI-PPPNet 的动机，作者仍视其为待验证能力 [23]。
- 有限秩近似的残差补偿是 LRP 留下的问题（DCRNetV2 为工程回答）[14]；参数化方法中角度
  误差随天线索引累积、比时延更敏感，是 HS-PINNnet 的分析结论与精度瓶颈 [15]。
- 一篇 2025 年的综述性文章指出，传统 CS 类方法依赖较强先验、在高压缩比下退化明显 [24]；
  这属于综述判断，不是某一篇原文的实验结论，引用时需注意其性质。

## 8. 参考文献

1. X. Rao, V. K. N. Lau. Distributed Compressive CSIT Estimation and Feedback for FDD
   Multi-User Massive MIMO Systems. IEEE Transactions on Signal Processing, 62(12):
   3261-3271, 2014. https://doi.org/10.1109/TSP.2014.2324991 ；预印本
   https://arxiv.org/abs/1405.2786
2. P. K. Gadamsetty, K. V. S. Hari, L. Hanzo. Learning a Common Dictionary for CSI Feedback
   in FDD Massive MU-MIMO-OFDM Systems. arXiv:2307.15943, 2023（venue 未核实）。
   https://arxiv.org/abs/2307.15943
3. P. Wu, Z. Liu, J. Cheng. Compressed CSI Feedback With Learned Measurement Matrix for
   mmWave Massive MIMO. arXiv:1903.02127, 2019（venue 未核实；正文自称 letter）。
   https://arxiv.org/abs/1903.02127
4. B. Groß, R. A. Salem, T. Wild, G. Wunder. Explicit CSI Feedback Compression via Learned
   Approximate Message Passing. International ITG Workshop on Smart Antennas (WSA), 2021.
   https://arxiv.org/abs/2110.05837
5. C.-K. Wen, W.-T. Shih, S. Jin. Deep Learning for Massive MIMO CSI Feedback. IEEE Wireless
   Communications Letters, 7(5): 748-751, 2018. https://arxiv.org/abs/1712.08919
6. J. Guo, C.-K. Wen, S. Jin, G. Y. Li. Convolutional Neural Network-Based Multiple-Rate
   Compressive Sensing for Massive MIMO CSI Feedback: Design, Simulation, and Analysis.
   IEEE Transactions on Wireless Communications, 19(4): 2827-2840, 2020.
   https://arxiv.org/abs/1906.06007
7. K. Gregor, Y. LeCun. Learning Fast Approximations of Sparse Coding. ICML, 2010.
   https://icml.cc/Conferences/2010/papers/449.pdf
8. A. Balatsoukas-Stimming, C. Studer. Deep Unfolding for Communications Systems: A Survey
   and Some New Directions. IEEE Workshop on Signal Processing Systems (SiPS), 2019.
   https://arxiv.org/abs/1906.05774
9. H. He, S. Jin, C.-K. Wen, F. Gao, G. Y. Li, Z. Xu. Model-Driven Deep Learning for Physical
   Layer Communications. IEEE Wireless Communications, 2019（卷期未核实）。
   https://arxiv.org/abs/1809.06059
10. J. Guo, L. Wang, F. Li, J. Xue. CSI Feedback with Model-Driven Deep Learning of Massive
    MIMO Systems. IEEE Communications Letters, 26(3): 547, 2022.
    https://arxiv.org/abs/2112.06405
11. Z. Hu, G. Liu, Q. Xie, J. Xue, D. Meng, D. Gündüz. A Learnable Optimization and
    Regularization Approach to Massive MIMO CSI Feedback. IEEE Transactions on Wireless
    Communications, 23(1): 104, 2024. https://arxiv.org/abs/2209.15340
12. Z. Liu, L. Wang, L. Xu, Z. Ding. Deep Learning for Efficient CSI Feedback in Massive MIMO:
    Adapting to New Environments and Small Datasets. IEEE Transactions on Wireless
    Communications, 23(9): 12297, 2024. https://arxiv.org/abs/2211.14785
13. Y. Ma, H. He, S. Song, J. Zhang, K. B. Letaief. Low-Complexity CSI Feedback for FDD
    Massive MIMO Systems via Learning to Optimize. IEEE Transactions on Wireless
    Communications, 24(4): 3483-3498, 2025. https://doi.org/10.1109/TWC.2025.3531447 ；
    预印本 https://arxiv.org/abs/2406.16323
14. S. Tang, Q. Yang, S.-W. Ko, J. Park, K. Huang. Model-Driven Deep Learning with Rank-One
    Sensing for Efficient CSI Feedback. arXiv:2610.11558, 2026.
    https://arxiv.org/abs/2610.11558
15. C. Ling, J. Guo, Y. Cui, C.-K. Wen, S. Jin, S. Han, X. Wang. Physics-Informed
    Path-Parametric Learning for Efficient and Lightweight CSI Feedback. arXiv:2606.23730,
    2026（注脚：较早版本被 IEEE ICC 2026 接收）。https://arxiv.org/abs/2606.23730
16. M. Tao, J. Fan, H. Xie, K. Xie. Deep CSI Feedback for FDD Massive MIMO Systems: A
    Curvelet Learning Approach. arXiv:2606.17737, 2026. https://arxiv.org/abs/2606.17737
17. Y. Song, T. Yang, K. Zhi, S. Li, F. Wu, S. Xue, G. Caire. Prior-Aided Masked Vector
    Quantization CSI Feedback for FDD Massive MIMO Systems. arXiv:2609.17769, 2026.
    https://arxiv.org/abs/2609.17769
18. M. Li, W. Xu, Z. Hu, A. Liu. Variable-Length Wideband CSI Feedback via Loewner
    Interpolation and Deep Learning. arXiv:2601.08300, 2026.
    https://arxiv.org/abs/2601.08300
19. Y. Cheng, F. Meng, J. Zou, J. Xie, Z. Zhang, Y. Wang, Y. Liang. Variable-Length
    Finite-Rate CSI Feedback With Generative Priors. IEEE Wireless Communications Letters
    （注脚注明已接收）, arXiv:2606.06846, 2026. https://arxiv.org/abs/2606.06846
20. D. Ulyanov, A. Vedaldi, V. Lempitsky. Deep Image Prior. CVPR, 2018；期刊版
    International Journal of Computer Vision, 2020. https://arxiv.org/abs/1711.10925
21. B. Vilas Boas, W. Zirwas, M. Haardt. Transfer Learning Capabilities of Untrained Neural
    Networks for MIMO CSI Recreation. arXiv:2111.07858, 2021（venue 未核实）。
    https://arxiv.org/abs/2111.07858
22. W. Wan, W. Chen, S. Wang, G. Y. Li, B. Ai. Deep Plug-and-Play Prior for Multitask Channel
    Reconstruction in Massive MIMO Systems. IEEE Transactions on Communications, 72, 2024.
    https://arxiv.org/abs/2308.04728
23. W. Chen, W. Wan, S. Wang, P. Sun, G. Y. Li, B. Ai. CSI-PPPNet: A One-Sided One-for-All
    Deep Learning Framework for Massive MIMO CSI Feedback. IEEE Transactions on Wireless
    Communications, 23(7): 7599-7611, 2024. https://doi.org/10.1109/TWC.2023.3342735
24. K. Zhang. AI-Driven Fronthaul Link Compression in Wireless Communication Systems: Review
    and Method Design. arXiv:2509.04805, 2025. https://arxiv.org/abs/2509.04805
25. S. Deka, K. Deka, N. T. Nguyen, S. Sharma, V. Bhatia, N. Rajatheva. Comprehensive Review
    of Deep Unfolding Techniques for Next-Generation Wireless Communication Systems.
    arXiv:2502.05952, 2025. https://arxiv.org/abs/2502.05952

## 9. 核查记录

**来源分级。**

- 原文全文：LORA [11]（ar5iv 全文，用于 CR 定义、1000 epoch/batch 200/lr 1e-3、
  0.0053 与 0.0069、编码器仅测量矩阵等）；LRP [14]（arXiv HTML 全文，用于 COST2100
  32×32 设置、100k/30k/20k 划分、1500 epoch、−13.18 dB 与 FISTA −10.46 dB、FLOPs）。
- 摘要：CsiNet+ [6] 的 38.0%/46.7% 参数缩减；Learned AMP [4] 的合成数据预训练与行稀疏
  MMV；FISTA-Net [10] 的 TSLR 与 ST-Net；SPTM2 [12] 的球面归一化与可训练测量矩阵；
  L2O [13] 的线性投影 + 可学习稀疏变换 + 共享 LSTM；HS-PINNnet [15] 的 92.8% FLOPs
  缩减、两数量级 FPGA 延迟、1% 角度误差近 50% 失真；PM-VQ [17] 的 Sionna 38.901 UMa
  与 diffuse 区域结论；SwinCANet [16] 的曲线波分解；Loewner [18] 的 U6G 频段与 tail
  dropout；CsiCoGen [19] 的全部 NMSE/rho 数值；Wan [22] 的三任务与 DeepMIMO；
  CSI-PPPNet [23] 的单侧框架与任意 CR 复用；Vilas Boas [21] 的未训练网络拟合与迁移；
  展开综述 [25] 的覆盖面（用于说明展开在检测/估计/波束等任务上的应用，未涉及 CSI 反馈）。
- 二手来源（仅用于书目信息或背景判断）：LORA 的卷期来自 NASA ADS 书目记录
  （2024ITWC...23..104H）；SPTM2 的卷期来自 ADS（2024ITWC...2312297L）；FISTA-Net 的
  卷期来自 ADS（2022IComL..26..547G）；CSI-PPPNet 的卷期与 DOI 来自出版社/索引页；
  Ma 等 [13] 的卷期与 DOI 来自出版方引用列表；CsiNet+ 的卷期来自多篇论文的参考文献
  列表；综述 [24] 关于 CS 高压缩比退化的判断属二手综述观点，未追溯到原始实验。

**不确定项。**

- [2][3][21] 的 venue 未核实，按 arXiv 预印本处理；[7] 为 ICML 2010 会议论文，页码未核实。
- [16][17][18][14][15][19] 均为 2026 年 arXiv 条目，[25] 为 2025 年 arXiv 综述；除 [15][19]
  注明被会议/期刊接收外，其余未见同行评议版本，结论应按预印本对待。
- 所核查论文几乎都未说明 checkpoint 选择策略，也未区分「训练中观察到的最优值」与
  「固定 checkpoint 复评值」；本文件因此不把这些数值当作可复现的定论。
- 本文件未核对各论文的随机种子与重复次数。

**检索词与日期。** 检索日期 2026-11-03。主要检索式：`all:"CSI feedback" AND
all:"compressed sensing"`；`all:"CSI feedback" AND all:"deep unfolding"`；
`all:"CSI feedback" AND all:"model-based"`；`all:"CSI feedback" AND all:"low-rank"`；
`all:"CSI feedback" AND all:prior`；`all:"CSI feedback" AND all:unfolding`；
`all:"approximate message passing" AND all:"CSI feedback"`；`all:"dictionary learning" AND
all:"channel feedback"`；`all:"untrained" AND all:"CSI"`；`all:"plug-and-play" AND all:CSI`；
`all:"sparsity" AND all:"CSI feedback"`；另按 `submittedDate` 倒序复查
`all:"CSI feedback"` 的 2025-2026 条目。venue 核对使用 WebSearch 检索出版社/索引页。
