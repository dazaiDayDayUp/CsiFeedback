# 时变与高移动场景下的 CSI 反馈

## 1. 范围与检索方法

本文件关注"信道随时间变化"这一维度下的下行 CSI 压缩反馈：时间相关性如何被建模、CSI 预测与反馈如何联合、反馈开销与延迟约束如何进入设计、压缩比能否可变、FDD/TDD 互易性假设如何被利用，以及多用户/多天线扩展。**不涉及具体新方法设计**。

需要先区分两类设定，后文的归类均以此为准：

- **设定 A：单快照静态 CSI。** 每个样本是一个独立信道实现，反馈链路只看到当前快照 `(2,32,32)`。本仓库的 CsiNet、CLNet、STNet 基线、以及 `csifeedback/data/cost2100.py` 的数据管线都属此类（数据加载不含帧/序列维度）。
- **设定 B：时变序列。** 样本是长度 `T` 的快照序列，编码器或解码器可访问历史。此时"压缩比"之外还多出两个自由度：每次反馈的比特数、以及反馈的**周期**（每隔多少帧报一次）。

这一区分很重要：设定 B 的论文常把"时间相关性增益"和"减少反馈次数"混在一起报告，其 NMSE 与本仓库设定 A 的 NMSE 不可直接比较。

检索在 arXiv API、Crossref、OpenAlex 上完成，检索词包括 `temporal CSI feedback`、`time-varying CSI feedback`、`CSI feedback delay`、`variable rate CSI feedback`、`CSI feedback reciprocity`、`multi-user CSI feedback`、`high mobility CSI feedback`，并对 2025-2026 年的 arXiv 结果做了单独筛查（检索日期见文末核查记录）。共保留 24 篇：2019-2020 年 6 篇、2021-2024 年 9 篇、2025-2026 年 9 篇。

## 2. 时间线与分类

按"时间信息进入系统的位置"可分为四类，这一划分比按网络结构划分更能说明各工作的实际假设：

1. **输入级多帧**：把连续 `T` 帧堆叠为额外通道或做 3D 卷积，由网络自行学习帧间关系。
2. **隐状态级循环**：LSTM/GRU/SSM 维护隐状态，反馈的是当前帧的隐表示。
3. **差分级**：只编码相对上一恢复帧的残差（或对残差做条件编码），解码端累积恢复。
4. **预测级**：显式预测未来帧，再决定反馈什么、何时反馈；预测器可放在 UE、BS 或两侧。

本次检索到的纯"输入级多帧"方案很少单独出现，通常是多帧输入与预测/循环结构结合，例如 [7] 的 3D CNN 直接以历史帧序列为输入。

时间线大致为：2018-2020 年出现把卷积自编码器与 LSTM 结合的早期方案 [1][2][3]，以及把延迟写进训练目标的做法 [8]；2020-2023 年转向"结构化地利用时间相关性"，出现马尔可夫差分编码 [4]、多分辨率结构 [13]、可变长度反馈 [14]；2024-2026 年则集中出现参数化信道模型 + 预测 [11]、生成式可变长反馈 [17][18]、上行辅助 + 预测 [10][22] 与多用户下的率选择 [24] 等系统级组合。

## 3. 代表性工作

### 3.1 从单快照到序列：时间相关性的显式建模

**[1] Wang et al., 2019（IEEE WCL）** 把 CsiNet 的编解码器与 LSTM 结合为 CsiNet-LSTM，直接从时变信道样本中同时学习空间结构与时间相关性，并称其在压缩比下降时仍保持稳健。这是"设定 A → 设定 B"的早期直接迁移，其代价是反馈内容从单帧潜向量变为依赖历史的潜向量。

**[2] Lu et al., 2019（IEEE Communications Letters）** 提出以 LSTM 模块利用信道的时频相关性来提升量化反馈精度，并给出参数量显著缩减的变体以折中复杂度与性能。与 [1] 的差别是它同时强调频域相关性，而不只强调帧间相关性。

**[3] Li & Wu, 2020（IEEE WCL）** 用深度循环网络学习时间相关性、用深度可分离卷积压缩模型，并比较了不同结构下"时空特征解耦"的表示方式，报告在低压缩比下更稳健。该工作把"时间建模"与"轻量化"放在同一目标下处理。

**[4] Liu, del Rosario & Ding, 2022（IEEE TWC）** 的 MarkovNet 以马尔可夫模型刻画信道演化，对前向 CSI 做**时间差分编码**，用 CNN 而非 RNN 实现，并报告在恢复精度和计算量上同时优于循环网络方案；文中还讨论了球面归一化与反馈量化等实现细节。它说明"时间相关性"不必依赖循环结构，显式差分 + 卷积即可获得增益。

**[5] Saini et al., 2025（arXiv）** 指出"循环网络 + 预训练自编码器"的组合会带来参数量与复杂度膨胀，提出参数量更低的循环自编码器以联合利用时-空-频（TSF）相关性。该文明确将低移动性下的强时间相关性作为前提。

**[6] Chatziloizos et al., 2026（arXiv）** 提出 k-记忆马尔可夫 VAE（k-MMVAE），在**潜空间**上用有限记忆的马尔可夫动力学建模 CSI 演化，而不是把多帧观测简单拼接。作者称其在低与中等压缩比下优于无记忆与"弱序列"基线，理由是显式潜动力学比隐式多帧输入更有效地利用了时间依赖。

### 3.2 预测与反馈联合：对抗延迟与 CSI 老化

这类工作把"反馈延迟"视为一阶问题：BS 用到的 CSI 在数值上已经对应过去的信道。

**[7] Zhang et al., 2022（arXiv）** 用 3D CNN 从历史信道序列直接预测未来下行 CSI，以抵消处理与反馈延迟，并报告相对"采样保持"（sample-and-hold）基线有明显改善。其设定是"预测器只看过去观测"，不改变反馈链路本身。

**[8] Jang et al., 2019（IEEE WCL）** 把反馈误差与反馈延迟一起建模进深度自编码器：训练时以**延迟后的下行信道**作为目标输出，使网络学到"补偿延迟"的映射。与推理阶段事后补偿不同，该文把延迟写进了训练目标。

**[9] Shehzad, Rose & Assaad, 2024（IEEE TWC）** 的 CSILaBS 反向处理：把主要机器学习负担放在 BS，UE 侧只用一个轻量预测函数评估反馈，从而在降低空口开销的同时避免 UE 端推理成本；多用户情形下还讨论了如何借助该预测函数挑选反馈内容。评估使用 Nokia Bell-Labs 的实测数据集。

**[10] He et al., 2025（arXiv）** 面向 FR3 TDD 系统，把上下行联合估计（JUDCEN）、量化反馈（TMCFN）与基于 Mamba 选择性状态空间模型的信道预测（MCPN）串成一条链路，在角-时延域捕捉长程时间动态。该文同时指出更高载频意味着更快的信道变化，因而预测环节不可省略。

**[11] Chang et al., 2026（arXiv）** 针对多用户 FDD：BS 先从 SRS 估计上行**信道参数**，UE 不显式估计下行信道，而是把观测编码为反馈符号；BS 解码后与部分互易的上行参数结合，预测**未来的**下行参数，再由显式信道模型合成信道矩阵。该文把"反馈什么"从信道矩阵改为物理参数，是 3.2 与 3.4 的交汇点。

**[12] Bondada, Jakubisin & Buehrer, 2025（arXiv）** 不提出反馈网络，而是解析地推导同步偏差与 CSI 反馈延迟对分布式 MIMO 容量的影响，指出时/频偏与反馈延迟会造成层间干扰、时偏还会造成符号间干扰。它为"延迟必须被显式建模"提供了分析侧证据。

### 3.3 可变与可配置压缩比

设定 B 下不同帧的可压缩性不同，固定 CR 因此未必最优：[14] 指出各用户的信道矩阵稀疏程度不同，[24] 指出固定比率方案难以适应非平稳信道。

**[13] Lu, Wang & Song, 2020（IEEE ICC）** 的 CRNet 通过多分辨率卷积提取特征，在相同计算复杂度下优于 CsiNet。其"多分辨率"结构与可变 CR 相关，但摘要只陈述精度结论，未给出多 CR 的训练/选择协议（未核实），本调研不把它计为已验证的可变率方案。

**[14] Nerini et al., 2023（IEEE TWC）** 用 PCA 把信道压到自适应维度的潜空间，按最小化 NMSE 的准则把比特分配到各潜维度，再用 k-means 确定量化码本，从而支持**变长**反馈。摘要报告：相对 CsiNetPro，迫零波束成形和速率提升 17%、参数量减少 23.4 倍、训练样本少 8 倍（数据集与训练预算未在摘要中说明）。

**[15] Park, Do & Lee, 2024（ICASSP）** 用带熵瓶颈（entropy bottleneck）的 VAE 实现多速率、变长压缩，摘要只给出"NMSE 优于现有 CSI 压缩方法"的结论，未给具体数值。

**[16] Lin, Lee & Ding, 2024（IEEE TWC）** 的 DCA 框架用分治结构同时适配不同天线端口数与不同反馈压缩级别，并让 UE 分段反馈以降低复杂度；其多速率逐次卷积编码器参数量少于 1000。该工作把"可变 CR"和"可变天线数"视为同一可扩展性问题。

**[17] Song et al., 2026（arXiv）** 的 PM-VQ 在角-时延功率图先验下把 CSI 映射为潜 token，用空间自适应掩码按预算挑选最重要 token，再对选中 token 做矢量量化并回传其位置；**单个模型**在一系列 token 数上训练，因而无需重训即可跨多个反馈维度工作。评估在 Sionna 生成的 3GPP TR 38.901 UMa 数据上，聚焦能量弥散的困难区间。

**[18] Cheng et al., 2026（arXiv）** 的 CsiCoGen 把有限比特决策移到反向扩散轨迹上的"码本约束高斯新息选择"，长度 `L` 的前缀对应 `R_L = L·log2 K` 比特，码本免训练且不需在线传输。摘要给出 COST2100 上的数值：792 比特时 indoor/outdoor NMSE 为 −28.58/−13.96 dB（rho 0.9964/0.9597），1592 比特时为 −30.72/−20.37 dB（rho 0.9967/0.9748）；比特数与 1/4–1/64 压缩比的换算关系摘要未说明。

### 3.4 FDD/TDD 与互易性假设

FDD 下上下行不互易，但**部分**统计/路径参数互易，可用来减少需要反馈的信息量。

**[19] Liu, Zhang & Ding, 2019（IEEE WCL）** 提出 DualNet-MAG 与 DualNet-ABS：BS 用可得的**上行** CSI 辅助恢复下行 CSI，两个网络分别利用幅度、以及实/虚部绝对值的双向相关性，从而在低速率反馈下显著降低载荷。该文是"上行辅助下行"这一思路的代表性早期工作。

**[20] Liu & Simeone, 2021（arXiv）** 的 HyperRNN 用超网络 + RNN 把上行信道的长时特征迁移到下行，联合利用部分互易与衰落过程的时间相关性，并报告在更低 NMSE 的同时缩短了导频长度。

**[21] Lin, Lee & Ding, 2024（IEEE TWC）** 把互易性与**部分阵列导频**结合：只对部分天线子阵列发 CSI-RS，配合历史 CSI 恢复全阵列，并设计可重构的 CSI-RS 放置方案以同时降低下行训练开销与反馈编码开销。其假设是"时间相干 + 部分互易"同时成立。

**[22] Liu et al., 2025（IEEE GLOBECOM）** 的 Dual-ImRUNet 面向**隐式**反馈（反馈特征向量而非完整矩阵），用双向相关增强模块把上下行特征向量投影到各自特征子空间以去冗余，并用输入格式对齐模块提升跨环境稳健性；摘要称相对当时最优方法降低 85% 反馈开销，且在未见环境上保持稳健。

### 3.5 多用户与多天线扩展

**[23] Mourya, Amuru & Kuchi, 2022（arXiv）** 指出现有 CSI 反馈模型几乎都在"单用户单信道"下训练，在多用户、各用户编码器不同的场景中失效；该文用多任务学习联合训练若干已有模型，并给出一个与本仓库基线直接相关的观察：CsiNet 与 STNet 联合训练时，CsiNet 的"性能"提升 39%，系统和速率提升 0.07 bps/Hz（"性能"具体指 NMSE 还是 rho，摘要未说明）。

**[24] Ansarifard et al., 2026（arXiv）** 在 BS 侧放一个 RL 控制器，在**预训练的多速率自编码器库**上为每个用户、每个时刻选择 CR，奖励同时包含 SINR/频谱效率、反馈开销与模型切换的计算成本。摘要报告：在不同用户数与基线平均后，反馈开销降低 53.4%、下行和速率提升 53.64%、NMSE 降低 22.38%；实验数据为高维时延域 CSI 数据集，非 COST2100。

## 4. 对比表

**表 1：时间信息进入系统的位置（按第 2 节分类）**

| 工作 | 类别 | 具体机制 | 时间信息在哪一侧使用 |
|---|---|---|---|
| [1] CsiNet-LSTM | 隐状态级 | 编解码器接 LSTM | 编码 + 解码 |
| [2] Lu et al. | 隐状态级 | LSTM 利用时频相关 | 编码 + 解码 |
| [3] Li & Wu | 隐状态级 | 深度 RNN + 可分离卷积 | 编码 + 解码 |
| [4] MarkovNet | 差分级 | 马尔可夫差分编码 + CNN | 编码 + 解码 |
| [5] SwinLSTM-AE | 隐状态级 | 轻量循环自编码器（TSF） | 编码 + 解码 |
| [6] k-MMVAE | 隐状态级 | 潜空间有限记忆马尔可夫动力学 | 编码 + 解码 |
| [7] Zhang et al. | 预测级 | 3D CNN 预测未来帧 | 仅预测器 |
| [11] Chang et al. | 预测级 | 上行参数 + 反馈符号 → 预测未来参数 | BS 侧 |
| [24] Ansarifard et al. | 率选择 | RL 在 AE 库上选 CR | BS 侧 |

**表 2：可变压缩比的实现机制**

| 工作 | 机制 | 是否单模型覆盖多率 | 评测数据（摘要可见范围） |
|---|---|---|---|
| [14] Nerini et al. | PCA 潜维 + 比特分配 + k-means 码本 | 是（变长） | 未说明 |
| [15] Park et al. | VAE + 熵瓶颈 | 是（多速率/变长） | 未说明 |
| [16] DCA | 分治 + 分段反馈 | 是（多率 + 多端口） | indoor/outdoor |
| [17] PM-VQ | token 掩码 + 矢量量化 + 位置回传 | 是（多 token 数） | Sionna 3GPP UMa |
| [18] CsiCoGen | 扩散轨迹上的码本新息选择 | 是（前缀长度） | COST2100 |
| [24] Ansarifard et al. | 多速率 AE 库 + RL 选择 | 否（模型库） | 时延域 CSI（非 COST2100） |

两张表都只并列**机制与设定**，不并列精度：各文的 CR 定义（比特数 / 潜维度 / 实值数）、帧率、数据集与训练预算不同，数值不可横向排序。

## 5. 小结（文献事实）

- 2018-2020 年的早期工作多把时间相关性当作"附加增益"（在卷积自编码器上拼接 LSTM），2020 年后逐渐被当作**结构性假设**：差分编码 [4]、潜空间动力学 [6]、显式预测 [7][11] 都要求信道演化满足特定模型。
- 反馈延迟从"误差项"变成"设计对象"：既可写入训练目标 [8]，也可在 BS 侧主动预测 [7][9][11]；[12] 从容量角度说明延迟不可忽略。
- 可变压缩比的主要实现路径有四类：潜维裁剪 + 比特分配 [14]、熵瓶颈 [15]、token/前缀截断 [17][18]、模型库切换 [24]；"单模型覆盖多率"与"系统级选率"是两种不同层面的问题。
- 互易性方面，本次检索的文献一致地把 FDD 互易性限定为**部分**互易（统计量、路径参数、特征子空间），并用上行 CSI 作为旁信息 [19][20][21][22]；其中没有工作声称 FDD 下可直接省略反馈。
- 多用户方向的公开工作明显少于单用户：本次检索到的多为 2022 年后，且常把"多用户"处理为多任务训练 [23] 或率分配 [24]，而非多用户联合编码。

## 6. 文献明确指出的开放问题

以下各条均对应原文摘要中的显式表述，不是本调研的推断：

1. **无记忆模型浪费了时间相关性。** [6] 指出无记忆压缩模型不利用相邻快照的强相关，而简单的时序扩展只是把多次观测拼在一起、没有显式建模潜变量动力学。
2. **时序方案的参数/复杂度代价。** [5] 指出"循环网络 + 预训练自编码器"的组合会显著增加参数量与计算量；[4] 也把 RNN 的计算与存储成本列为部署障碍。
3. **率灵活性与比特接口的耦合。** [18] 指出已有的可扩展方案通过排序、掩码、量化、矢量量化或熵编码获得率灵活性，从而把有限比特接口与任务特定的潜编码器绑定在一起。
4. **固定压缩比无法适应非平稳信道。** [24] 指出传统固定比率方案难以适应非平稳信道条件，因而需要按系统状态动态选率。
5. **反馈延迟与同步误差的定量影响。** [12] 给出延迟/同步偏差导致层间干扰与符号间干扰的解析结论，但该分析针对分布式 MIMO 容量，未覆盖压缩反馈的端到端链路。

本次检索范围内，未见在 COST2100 上同时报告 1/4–1/64 全压缩比矩阵、并明确区分"训练中观察最优"与"checkpoint 复评"的时变工作；这只是检索范围的观察，不作为领域结论。

## 7. 参考文献

[1] T. Wang, C.-K. Wen, S. Jin, G. Y. Li. Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels. IEEE Wireless Communications Letters, 2019. DOI: 10.1109/LWC.2018.2874264. https://arxiv.org/abs/1807.11673

[2] C. Lu, W. Xu, H. Shen, J. Zhu, K. Wang. MIMO Channel Information Feedback Using Deep Recurrent Network. IEEE Communications Letters, 2019. DOI: 10.1109/LCOMM.2018.2882829. https://arxiv.org/abs/1811.07535

[3] X. Li, H. Wu. Spatio-Temporal Representation With Deep Neural Recurrent Network in MIMO CSI Feedback. IEEE Wireless Communications Letters, 2020. DOI: 10.1109/LWC.2020.2964550. https://arxiv.org/abs/1908.07934

[4] Z. Liu, M. del Rosario, Z. Ding. A Markovian Model-Driven Deep Learning Framework for Massive MIMO CSI Feedback. IEEE Transactions on Wireless Communications, 2022. DOI: 10.1109/TWC.2021.3103120. https://arxiv.org/abs/2009.09468

[5] A. Saini, Y. Xing, J. H. Kim, A. Ahmadian Tehrani, W. Gerstacker. SwinLSTM Autoencoder for Temporal-Spatial-Frequency Domain CSI Compression in Massive MIMO Systems. arXiv preprint, 2025. https://arxiv.org/abs/2505.04432

[6] E. Chatziloizos, K. Vandikas, A. Vulgarakis Feljan, Z. Chen, N. Pappas. Context-Aware Markov VAE for CSI Compression in Wireless Systems. arXiv preprint, 2026. https://arxiv.org/abs/2606.16607

[7] Y. Zhang, A. Alkhateeb, P. Madadi, J. Jeon, J. Cho, C. Zhang. Predicting Future CSI Feedback for Highly-Mobile Massive MIMO Systems. arXiv preprint, 2022. https://arxiv.org/abs/2202.02492

[8] Y. Jang, G. Kong, M. Jung, S. Choi, I. Kim. Deep Autoencoder Based CSI Feedback With Feedback Errors and Feedback Delay in FDD Massive MIMO Systems. IEEE Wireless Communications Letters, 2019. DOI: 10.1109/LWC.2019.2895039. https://doi.org/10.1109/LWC.2019.2895039

[9] M. K. Shehzad, L. Rose, M. Assaad. Massive MIMO CSI Feedback Using Channel Prediction: How to Avoid Machine Learning at UE? IEEE Transactions on Wireless Communications, 2024. DOI: 10.1109/TWC.2024.3376633. https://arxiv.org/abs/2403.13363

[10] X. He, H. Hou, Y. Wang, W. Wang, S. Jin, S. Chatzinotas, B. Ottersten. Deep Learning-Based Joint Uplink-Downlink CSI Acquisition for Next-Generation Upper Mid-Band Systems. arXiv preprint, 2025. https://arxiv.org/abs/2512.02557

[11] H. Chang, M. Nie, S. Li, C. Liu, B. Vucetic, Y. Li, W. Hardjawana. UL-Assisted Parametric CSI Feedback and Prediction for FDD Systems. arXiv preprint, 2026. https://arxiv.org/abs/2610.14649

[12] K. S. Bondada, D. Jakubisin, R. M. Buehrer. Impact of Synchronization Offsets and CSI Feedback Delay in Distributed MIMO Systems. arXiv preprint, 2025. https://arxiv.org/abs/2503.24314

[13] Z. Lu, J. Wang, J. Song. Multi-Resolution CSI Feedback With Deep Learning in Massive MIMO System. IEEE ICC, 2020. DOI: 10.1109/ICC40277.2020.9149229. https://arxiv.org/abs/1910.14322

[14] M. Nerini, V. Rizzello, M. Joham, W. Utschick, B. Clerckx. Machine Learning-Based CSI Feedback With Variable Length in FDD Massive MIMO. IEEE Transactions on Wireless Communications, 2023. DOI: 10.1109/TWC.2022.3215104. https://arxiv.org/abs/2204.04723

[15] B. Park, H. Do, N. Lee. Multi-Rate Variable-Length CSI Compression for FDD Massive MIMO. IEEE ICASSP, 2024. DOI: 10.1109/ICASSP48485.2024.10448212. https://arxiv.org/abs/2311.18172

[16] Y.-C. Lin, T.-S. Lee, Z. Ding. A Scalable Deep Learning Framework for Dynamic CSI Feedback With Variable Antenna Port Numbers. IEEE Transactions on Wireless Communications, 2024. DOI: 10.1109/TWC.2023.3305546. https://arxiv.org/abs/2204.09169

[17] Y. Song, T. Yang, K. Zhi, S. Li, F. Wu, S. Xue, G. Caire. Prior-Aided Masked Vector Quantization CSI Feedback for FDD Massive MIMO Systems. arXiv preprint, 2026. https://arxiv.org/abs/2609.17769

[18] Y. Cheng, F. Meng, J. Zou, J. Xie, Z. Zhang, Y. Wang, Y. Liang. Variable-Length Finite-Rate CSI Feedback With Generative Priors. arXiv preprint, 2026. https://arxiv.org/abs/2606.06846

[19] Z. Liu, L. Zhang, Z. Ding. Exploiting Bi-Directional Channel Reciprocity in Deep Learning for Low Rate Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 2019. DOI: 10.1109/LWC.2019.2898662. https://doi.org/10.1109/LWC.2019.2898662

[20] Y. Liu, O. Simeone. HyperRNN: Deep Learning-Aided Downlink CSI Acquisition via Partial Channel Reciprocity for FDD Massive MIMO. arXiv preprint, 2021. https://arxiv.org/abs/2104.12274

[21] Y.-C. Lin, T.-S. Lee, Z. Ding. Exploiting Partial FDD Reciprocity for Beam-Based Pilot Precoding and CSI Feedback in Deep Learning. IEEE Transactions on Wireless Communications, 2024. DOI: 10.1109/TWC.2023.3289929. https://arxiv.org/abs/2205.14558

[22] Z. Liu, Y. Ma, R. Tafazolli, Z. Ding. Leveraging Bi-Directional Channel Reciprocity for Robust Ultra-Low-Rate Implicit CSI Feedback with Deep Learning. IEEE GLOBECOM, 2025. DOI: 10.1109/GLOBECOM59602.2025.11432666. https://arxiv.org/abs/2507.12301

[23] S. Mourya, S. Amuru, K. K. Kuchi. Multi-Task Learning for Multi-User CSI Feedback. arXiv preprint, 2022. https://arxiv.org/abs/2211.08173

[24] M. Ansarifard, M. K. Sharma, G. Exarchakos, K. C. Joshi. System-Aware Adaptive CSI Feedback via RL-Guided Autoencoder Switching in Multi-User MIMO System. arXiv preprint, 2026. https://arxiv.org/abs/2607.25588

## 8. 核查记录

**检索日期**：2026-11-03。工具：arXiv API（`export.arxiv.org/api/query`）、Crossref REST API、OpenAlex API、`FetchURL`。检索词见第 1 节；对 2025-2026 年结果按 `published` 日期单独筛查。

**标题/作者/年份/venue 的核实方式**：全部 24 篇均通过 arXiv API 或 Crossref 的机读元数据核对，未依据记忆填写。venue 与 DOI 以 Crossref 记录为准（[1][2][3][4][8][9][13][14][15][16][19][21][22] 共 13 篇有 DOI 记录）；其余 11 篇为 arXiv 预印本，venue 记作"arXiv preprint"。

**摘要级信息（未读全文）**：第 3 节所有方法描述与结论均来自 arXiv/OpenAlex 摘要。以下数值来自摘要，未在全文复核，且未做协议对齐：

- [14]：+17% 迫零和速率、参数减少 23.4 倍、训练样本减少 8 倍；数据集、CR、训练预算摘要未给。
- [18]：COST2100 上 792/1592 比特的 NMSE 与 rho；比特数到 CR 的换算、训练 epoch、是否 checkpoint 复评均未在摘要说明。
- [22]：反馈开销降低 85%（相对"state-of-the-art method"，具体基线未指名）。
- [23]：39% 性能提升、+0.07 bps/Hz；"性能"指标未在摘要明确。
- [24]：−53.4% 反馈开销、+53.64% 和速率、−22.38% NMSE；为跨用户数与跨基线的平均值，数据集非 COST2100。

**未纳入的数值**：[1][2][3][4][5][6][7][9][10][11][12][13][15][16][17][19][20][21] 的摘要未给出可直接引用的 NMSE/rho，故本文不记录任何该类数字；这不表示原文没有数值。

**二手来源**：[23] 中提到 STNet 与本仓库基线同名，本文只按该文摘要陈述"CsiNet 与 STNet 联合训练"这一事实，未假定其 STNet 实现与本仓库 `csifeedback/models/stnet.py` 一致。其余条目未使用综述或博客数字。

**存疑/未核实项**：

1. [7] 的 arXiv 版本为 2022 年；Crossref 另有一条 2025 年 IEEE MECOM 的同名记录（DOI 10.1109/MECOM67453.2025.11439593），本次未核实两版本内容是否一致，故正文按 arXiv 2022 引用。
2. [13] 的"多分辨率"是否等价于"单模型支持多压缩比"，摘要未说明；正文已标注未核实，未计入表 2。
3. [5] 标题含 SwinLSTM，但摘要只描述"参数轻量的循环自编码器"，未说明 Swin 结构的具体作用；正文按摘要措辞描述。
4. [17] 的评测为 Sionna 生成的 3GPP TR 38.901 UMa 数据而非 COST2100，与其他条目不可直接比较。
5. [11] 与 [10] 均涉及"参数化/预测 + 反馈"，但一为 FDD 多用户、一为 FR3 TDD，协议不同，正文未做优劣比较。
