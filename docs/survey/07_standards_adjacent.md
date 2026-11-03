# 标准化与相邻任务：3GPP 码本、AI/ML 空口与 CSI 压缩的接口

本文只回答三个问题：3GPP 如何定义 CSI 反馈的码本与开销；AI/ML 空口标准化如何界定 CSI 压缩的模型边界与评估协议；物理层自编码器、联合导频/反馈、预编码接口这些相邻任务，与本仓库 COST2100 自编码器设定存在哪些结构性差异。文中不评价方法优劣，也不做跨协议排名。

## 1. 范围与检索方法

主题边界：3GPP NR 的 CSI 码本（Type I、Type II、eType II 及其端口选择变体、多 TRP 相干联合传输扩展）、AI/ML 空口研究项与工作项、物理层端到端自编码器、联合导频/信道估计/反馈/预编码，以及 CSI 压缩与预编码、波束管理的接口。不覆盖信道编码、调制识别、定位算法本身。

检索于 2026-11-03 进行，使用 arXiv API 与 WebSearch，检索词包括 `3GPP TR 38.843`、`Type II codebook`、`eType II`、`FeCoMP`、`CSI compression coherent joint transmission`、`two-sided CSI compression`、`joint pilot design CSI feedback`、`precoding-oriented CSI feedback`、`AI native air interface 3GPP`、`CSI codebook deep learning comparison` 等；时间范围 2018-01 至 2026-11。来源优先级为规范原文与论文原文 > 作者主页/出版社页面 > 综述；凡转引均在下文标注。

## 2. 两条标准化脉络

### 2.1 码本：从 Type I 到多 TRP 联合传输

3GPP 的 CSI 反馈是**隐式**的：UE 不上报信道矩阵，而是上报 RI、PMI、CQI，PMI 指向收发双方共享码本中的一个预编码矩阵 [2]。Rel-15 定义 Type I（单波束 DFT 向量按宽带报告，极化间共相位按子带报告，开销最小）与 Type II（L∈{2,3,4} 个波束的线性组合，幅度可宽带或子带报告、相位按子带报告，开销显著增大）[3][4]。Rel-16 的 eType II 引入频域压缩：以 M_v = p_v·N_3/R 个频域基压缩 N_3 个子带的系数（p_v∈{1/4, 1/8}），并用 β∈{1/4, 1/2, 3/4} 控制非零系数比例，因此 PMI 同时携带空域波束指示、频域基指示与量化系数 [3]。Rel-16 另有 Enhanced Type II 端口选择码本，Rel-17 的 FeType II PS 借助上下行部分互易性进一步降低端口选择开销 [3]。

"FeCoMP" 一名需要澄清。在 3GPP 语境中它是 LTE Rel-15 的 further enhanced CoMP 工作项，为 LTE 引入有限的多点非相干联合传输（NC-JT）支持；同一文献同时指出，当时 3GPP 尚无引入相干联合传输（C-JT）的计划 [5]。NR 侧的多 TRP 相干联合传输码本则是在 Rel-18 把 eType II 的空间-频率压缩框架跨 TRP 扩展，相关码本与报告配置见 TS 38.214 的 Rel-18 版本 [2]。因此把 FeCoMP 当作 NR 码本名称并不准确，文献中更常见的是 eType2CJT / feType2CJT 一类能力命名。

### 2.2 AI/ML 空口：从研究项到协议框架

Rel-18 研究项 TR 38.843 确定三个用例：CSI 反馈、波束管理、定位；其中 CSI 下分空频域压缩与时域预测两个子用例 [1][7]。CSI 压缩采用"双向模型"：UE 侧编码器与 gNB 侧解码器分离部署。压缩对象可以是 UE 估计的原始信道矩阵，也可以是由它导出的预编码矩阵；后者与既有码本框架兼容，因而在 3GPP 讨论中更受关注 [7]。Guo 等把这类方案归纳为单侧细化、双向隐式增强与双向显式增强三类框架，并指出显式反馈虽可能带来更大增益，但会改变系统使用 CSI 的方式，因而难以与现网反馈策略直接比较 [24]。

标准化真正棘手的是训练与互操作。TR 讨论了三类训练：Type 1 在一侧联合训练编码器与解码器后传输模型；Type 2 两侧在同一训练循环中交换前向激活与反向梯度；Type 3 两侧分离训练，再通过共享数据对齐 [7]。三者在性能、专有信息暴露与多厂商兼容之间取舍，Type 3 被认为更可行 [7]。此外，标准化还需处理模型泛化/切换/更新、性能监控与失败回退 [7]。

Rel-19 把 Rel-18 的部分结论转为规范工作（NR_AIML_air WI），覆盖 CSI 预测、波束管理与定位，核心空口工作于 2025 年完成；据一篇 2025 年综述，Rel-20 进一步讨论双向 CSI 压缩与跨厂商模型运行 [8]。2026 年的协议框架讨论主张：标准只定义 AI 功能的配置、验证、激活、监控与回退语义，不规定模型结构、训练方法或权重 [9]。这一立场与学术界"提出新网络结构"的取向有明显距离。本调研判断：这是标准化文档与学术论文在目标函数上的根本差异，而非技术路线之争。

### 2.3 评估协议：为什么 COST2100 结果不能直接与码本比较

按照 Guo 等的综述，3GPP 的 AI CSI 评估以 TR 38.901 的 CDL 信道与系统级仿真为基础，并区分"中间 KPI（GCS/SGCS）"与"最终 KPI（吞吐、开销）"两层指标 [6]。Guo 等的综述明确写道：现有多数工作基于 COST2100，该数据集"简单且远离实际系统"，**不能用于把 AI 反馈与现有 Type I/II 码本比较**；应改用 3GPP 约定信道并做系统级评估 [6]。这是本文后续所有差异讨论的出发点。

## 3. 代表性工作

### 3.1 物理层端到端自编码器

O'Shea 与 Hoydis 把发射机、信道、接收机整体视为一个自编码器，用可微信道模型端到端训练，学习到接近 QAM 的星座与脉冲成形 [10]。Ait Aoudia 与 Hoydis 把这一思路推到 OFDM，并展示"无导频"通信：接收机直接从数据符号学习均衡，从而省去导频 [11]。这类工作的共同点是信道位于训练环内、损失为链路级指标（BER/BLER/速率），不存在"有限比特瓶颈 + 两侧模型归属不同厂商"的约束。3GPP 综述也把"完全端到端自编码器"归入 6G 以后才讨论的范畴 [6]。

### 3.2 联合导频、信道估计与反馈

Sohrabi 等把多用户信道估计与反馈视为分布式信源编码问题：联合设计导频，并用一个 DNN 把接收导频直接映射为反馈比特、再把所有用户的比特映射为预编码矩阵；短导频与有限反馈下已接近全 CSI 线性预编码 [12]。Guo 等构造显式与隐式两套联合估计-反馈网络，显式网络用多 SNR 训练加深度残差重建，隐式网络直接压缩导频，并加入量化模块生成比特流 [13]。Carpi 等把导频、UE 侧压缩器与基站处理一起训练，损失直接写为可达和速率，使反馈目标与预编码目标对齐 [14]。三者都把"CSI 获取"视为一个跨模块的端到端问题，而不是独立的压缩问题。

### 3.3 与码本、预编码接口的工作

Chen 等的隐式反馈用神经网络替换 PMI 编解码：单 RB 下相对 Type I 码本节省 25.0%/40.0% 开销（两种天线配置），52 RB 宽带下相对 Type II 码本节省 30.7%/48.0%（不计/计入子带相关性）[15]。Liu 等的 EVCsiNet 在 3GPP 链路级信道上反馈特征向量，属于"码本兼容"的中间路线 [16]。Ma 等把深度学习嵌入 Rel-17 Type II：用 focal loss 精化角-时延域端口选择，再在 gNB 侧重建，并以 MSE 与和速率的两阶段损失训练；摘要报告和速率相对 R17 Type II 与传统深度学习基线提升 10% 以上 [17]。

Luo 等提出"标准编码器 + 站点特定生成解码器"：UE 侧沿用标准压缩，gNB 侧用生成模型细化，以规避双向模型的标准化与互操作难题，并降低数据采集成本 [18]。Gouda 等改为反馈"有效 CSI"（UE 用预编码导频计算合并器后的有效信道），保持与传统方案相同的载荷结构，仿真报告平均和速率提升至多 30%，干扰受限时第 10 百分位每流速率增益可达 100% [19]。

2026 年的 CsiCoGen 把有限比特决策移到"码本约束的高斯创新选择"上，码本免训练且不需在线传输；在 COST2100 上报告 indoor/outdoor NMSE 为 −28.58/−13.96 dB（792 bits）与 −30.72/−20.37 dB（1592 bits），rho 为 0.9964/0.9597 与 0.9967/0.9748 [20]。该文同时给出比特预算与 COST2100 指标，但其开销以比特而非压缩比计，不能直接换算为本仓库的 CR。

值得注意的是反向证据：Praneeth 等在同一类下行 CSI 压缩任务上比较 PCA 与常用 DNN，摘要称 PCA 取得了与 DNN"可比"的重建性能，且不存在模型复杂度、跨场景泛化与厂商兼容问题 [21]。该文为预印本，未给出与本仓库可对齐的数值，但提示"深度模型必然更优"并非已被普遍证实的结论。

### 3.4 波束管理：另一种反馈对象

Jayaweera 等在 3GPP 系统级仿真中集成空域/时域波束预测模型，评估波束预测精度、泛化与用户吞吐，并给出轻量化设计准则 [22]。其反馈对象是 L1-RSRP 测量与 top-K 波束索引，而非 2048 维 CSI；因此它优化的是测量/报告开销与波束选择时延，与 CSI 压缩不是同一任务；两者的收益来源不同，不宜互相替代论证。

## 4. 对比表

表 1 按协议维度对照四种设定。表中"本仓库"一列仅描述当前配置，不代表优劣。

| 维度 | 本仓库 COST2100 自编码器 | 3GPP Type II / eType II | 3GPP Rel-18 AI/ML CSI 压缩 | 物理层端到端自编码器 |
|------|--------------------------|--------------------------|-----------------------------|----------------------|
| 反馈对象 | 显式 CSI，`(2,32,32)` | 隐式预编码器（PMI） | 显式信道或预编码矩阵 | 发射符号/波形 |
| 开销单位 | 压缩比 CR = 2048/瓶颈维度 | PMI 比特数 | 比特数（含量化） | 符号数/带宽 |
| 信道与评估 | COST2100，链路级 | TR 38.901 CDL，系统级 | TR 38.901 CDL，系统级 | 可微信道模型或实测 |
| 主要指标 | NMSE(dB)、rho | GCS/SGCS、吞吐 | SGCS + 吞吐/开销 | BER/BLER、速率 |
| 两侧关系 | 联合训练，单一实现 | 无训练，共享码本查表 | 双向模型，需配对与生命周期管理 | 联合训练，同一实现 |
| 标准化状态 | 学术基准 | Rel-15/16/17 规范 | Rel-18 研究，Rel-19 部分规范化 | 未标准化 |

表 2 说明为什么两类指标不能互相换算。

| 项目 | 本仓库设定 | 3GPP 码本/AI 评估 | 不可直接换算的原因 |
|------|------------|-------------------|--------------------|
| 相似度 | rho：重建 CSI 与真实 CSI 的余弦相似度 | GCS/SGCS：预编码器或空频域 CSI 的余弦相似度 | 比较对象不同（信道 vs 预编码器），聚合方式也不同 |
| 开销 | CR 只约束瓶颈维度，多数工作不含量化 | 比特数，含量化与报告配置 | 同一 CR 在不同量化位数下对应不同比特数 |
| 性能 | NMSE 与 rho | 吞吐/BLER，辅以 GCS | NMSE 与和速率非单调；预编码对误差方向敏感 |
| 模型选择 | baseline 用 `select_best: "all"`（test 指标） | 约定 KPI 与固定测试流程 | 选择策略不同会抬高报告值 |

## 5. 与本仓库设定的结构性差异

1. **反馈对象不同。** 本仓库重建 `(2,32,32)` 的显式 CSI，再由此计算预编码；3GPP 码本反馈的是预编码器本身。前者保留全部信道信息但开销大，后者只保留与预编码相关的自由度。综述指出，显式反馈"无法与现有系统的反馈策略直接比较" [6]。
2. **开销语义不同。** CR=1/4 表示瓶颈为 512 个实数，但若每个实数量化到 b 比特，实际上行比特数为 512b；文献中大量工作默认无量化或仅做仿真量化。CsiCoGen 以 792/1592 bits 报告结果 [20]，正说明比特与 CR 之间需要额外假设才能互换。
3. **信道与场景不同。** COST2100 是统计信道数据集，3GPP 评估用 TR 38.901 CDL 并叠加调度、多用户干扰与反馈周期。前者适合比较重建网络，后者才回答"能否替代码本"。
4. **指标不同。** NMSE/rho 衡量重建保真度；系统关心的是吞吐与 BLER。Carpi 等直接把损失写成和速率 [14]，Gouda 等改喂有效 CSI [19]，都是对这一错配的回应。
5. **训练与部署假设不同。** 本仓库两侧联合训练、单一模型、无模型传输与监控；3GPP 框架需处理跨厂商配对、模型生命周期管理、性能监控与回退 [7][8]。Type 3 分离训练之所以更可行，正是为了不让一家厂商的重训牵动全生态 [7]。
6. **预编码约束不同。** 本仓库只评估 CSI 重建，不引入码本、rank 或 PMI 负载约束；若用于预编码，可自由选择 SVD 等方案。码本方案受码本结构与量化约束，AI 隐式方案受 PMI 负载结构约束 [15][17]。

## 6. 小结（仅陈述文献事实）

- 3GPP 的 CSI 反馈是隐式、以比特计、受码本约束的；AI/ML 标准化的主线是把神经网络嵌入这一框架（多为压缩预编码矩阵），而不是替换为显式信道自编码器 [6][7]。
- 已进入标准化议程的 AI CSI 功能以"压缩"与"预测"两类为主；双向模型的训练方式、模型配对与监控是标准化核心难点 [1][7][8]。
- 学术主流仍是"显式 CSI + NMSE/rho + COST2100"，与 3GPP 评估协议存在系统性差异；综述明确指出基于 COST2100 的结果不能与码本比较 [6]。
- 相邻任务（物理层自编码器、联合导频、有效 CSI、预编码导向损失）共同说明：压缩误差最小化与链路性能最大化不是同一目标 [11][14][19]。

## 7. 文献明确指出的开放问题

- **评估公平性**：需要 3GPP 信道与系统级评估，COST2100 不足以支撑与码本的比较 [6]。
- **复杂度与推理时延**：FLOPs 不能完全代表推理速度，还需考虑访存与并行度；UE 侧模型必须轻量 [6]。
- **泛化与环境漂移**：模型切换、更新与监控机制；数字孪生数据能否迁移到真实环境仍需验证，有实测工作指出即便配合在线学习，专用数字孪生仍不可少 [8][23]。
- **互操作与模型配对**：编码器/解码器跨厂商兼容、模型传输与责任划分 [7][8]。
- **有限比特接口**：量化、变速率与熵编码如何与学习到的潜变量耦合 [20]。
- **指标错配**：NMSE 与吞吐/和速率不一致，任务导向的损失与评价仍不统一 [14][19]。
- **码本侧的老问题**：高移动、多 TRP 与超大规模 MIMO 对码本结构与反馈开销的挑战 [3]。

## 参考文献

[1] 3GPP. Study on Artificial Intelligence (AI)/Machine Learning (ML) for NR Air Interface. TR 38.843 V18.0.0, Dec. 2023. https://www.3gpp.org/dynareport/38843.htm

[2] 3GPP. NR; Physical Layer Procedures for Data. TS 38.214 (Rel-18). https://www.3gpp.org/dynareport/38214.htm

[3] Z. Qin and H. Yin. A Review of Codebooks for CSI Feedback in 5G New Radio and Beyond. China Communications, 22(2):112–127, 2025. https://doi.org/10.23919/JCC.ja.2023-0117 （预印本：https://arxiv.org/abs/2302.09222）

[4] E. Onggosanusi et al. Modular and High-Resolution Channel State Information and Beam Management for 5G New Radio. IEEE Communications Magazine, 56(3):48–55, 2018. https://doi.org/10.1109/MCOM.2018.1700761

[5] S. Muruganathan et al. On the System-Level Performance of Coordinated Multi-Point Transmission Schemes in 5G NR Deployment Scenarios. arXiv:1906.07252, 2019.（venue 未核实）https://arxiv.org/abs/1906.07252

[6] J. Guo et al. AI for CSI Feedback Enhancement in 5G-Advanced. IEEE Wireless Communications, 31(3):169–176, 2024. https://arxiv.org/abs/2206.15132

[7] X. Lin. An Overview of the 3GPP Study on Artificial Intelligence for 5G New Radio. arXiv:2308.05315, 2023.（venue 未核实）https://arxiv.org/abs/2308.05315

[8] C.-H. Huang et al. AI/ML Life Cycle Management for Interoperable AI Native RAN. arXiv:2507.18538, 2025.（venue 未核实）https://arxiv.org/abs/2507.18538

[9] X. Lin. Toward AI-Native 6G Air Interface: A 3GPP Perspective on Protocol Framework. arXiv:2606.27466, 2026.（venue 未核实）https://arxiv.org/abs/2606.27466

[10] T. O'Shea and J. Hoydis. An Introduction to Deep Learning for the Physical Layer. IEEE Transactions on Cognitive Communications and Networking, 3(4):563–575, 2017. https://doi.org/10.1109/TCCN.2017.2758370

[11] F. Ait Aoudia and J. Hoydis. End-to-End Learning for OFDM: From Neural Receivers to Pilotless Communication. IEEE Transactions on Wireless Communications, 21(2):1049–1063, 2022. https://doi.org/10.1109/TWC.2021.3101364

[12] F. Sohrabi, K. M. Attiah, and W. Yu. Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD Massive MIMO. IEEE Transactions on Wireless Communications, 20(7):4044–4057, 2021. https://doi.org/10.1109/TWC.2021.3055202

[13] J. Guo et al. Deep Learning for Joint Channel Estimation and Feedback in Massive MIMO Systems. Digital Communications and Networks, 10(1):83–93, 2024. https://arxiv.org/abs/2011.07242

[14] F. Carpi et al. Precoding-Oriented Massive MIMO CSI Feedback Design. IEEE ICC 2023, pp. 4973–4978. https://arxiv.org/abs/2302.11526

[15] M. Chen et al. Deep Learning-Based Implicit CSI Feedback in Massive MIMO. IEEE Transactions on Communications, 70(2):935–950, 2022. https://arxiv.org/abs/2105.10100

[16] W. Liu et al. EVCsiNet: Eigenvector-Based CSI Feedback Under 3GPP Link-Level Channels. IEEE Wireless Communications Letters, 10(12):2688–2692, 2021. https://doi.org/10.1109/LWC.2021.3112747

[17] K. Ma et al. Deep Learning Empowered Type-II Codebook: New Paradigm for Enhancing CSI Feedback. arXiv:2305.08081, 2023.（venue 未核实）https://arxiv.org/abs/2305.08081

[18] H. Luo, S. R. Khosravirad, and A. Alkhateeb. Generative Decoding of Compressed CSI for MIMO Precoding Design. arXiv:2511.07783, 2025. https://arxiv.org/abs/2511.07783

[19] B. Gouda et al. Codebook-Based Effective CSI Feedback for Precoding Design in MIMO Systems. arXiv:2609.27505, 2026. https://arxiv.org/abs/2609.27505

[20] Y. Cheng et al. Variable-Length Finite-Rate CSI Feedback With Generative Priors. arXiv:2606.06846, 2026. https://arxiv.org/abs/2606.06846

[21] K. S. Praneeth et al. Study on Downlink CSI Compression: Are Neural Networks the Only Solution? arXiv:2502.17459, 2025. https://arxiv.org/abs/2502.17459

[22] N. Jayaweera et al. 5G-Advanced AI/ML Beam Management: Performance Evaluation with Integrated ML Models. arXiv:2404.15326, 2024.（venue 未核实）https://arxiv.org/abs/2404.15326

[23] T.-H. Huang et al. AI-Based CSI Feedback with Digital Twins: Real-World Validation and Insights. arXiv:2505.00660, 2025. https://arxiv.org/abs/2505.00660

[24] Y. Guo et al. Deep Learning for CSI Feedback: One-Sided Model and Joint Multi-Module Learning Perspectives. IEEE Communications Magazine, 63(7):90–97, 2025. https://doi.org/10.1109/MCOM.001.2400285

## 核查记录

**来源等级**

- 原文全文或相关段落：[3]（码本结构、L/M_v/β 参数）、[5]（feCoMP 与 NC-JT 的原句）、[6]（三类框架、GCS 表、评估与开放问题段落）、[7]（三个用例、三种训练方式、协作层级）。以上引用内容逐句核对过 PDF 文本。
- 仅读摘要：[8]–[24]（其中 [10]–[15] 也只读了摘要）。结论只写到摘要明确陈述的范围。
- 仅核对题录（标题、作者、卷期、DOI，未读摘要）：[4]。
- 规范条目：[1][2] 的标题、版本与日期核对到 3GPP dynareport 页面（可访问，HTTP 200），但未逐条阅读规范正文条款；CJT 码本的描述依据 TS 38.214 Rel-18 的公开版本与综述转述，条款号未核实。
- 二手来源：[6] 表 I 中 Type II 码本 300 bits / GCS 0.9042、AI 反馈 104 bits / GCS 0.9044 等数值，是该综述转引 3GPP 评估贡献的结果，本调研未取得原始贡献文档，按二手引用处理；同理，"Type I 码本 GCS 仅 0.76"与"约 300 bits"的说法也来自该综述。
- 未核实项：[5][7][8][9][17][21][22] 的正式 venue；[3] 中关于 Rel-18 CJT 能力命名的细节（该综述未覆盖 Rel-18，命名来自规范文本检索与二手引用）。[3][12][16][24] 的卷期页码与 DOI 经 Crossref 或出版社页面核对。

**数值记录边界**

- 本文共记录四处性能数值：Chen 等的开销节省比例 [15]、Ma 等的和速率提升 10% 以上 [17]、Gouda 等的 30%/100% [19]、CsiCoGen 的 NMSE/rho 与比特数 [20]；均取自摘要，并附带了数据集与开销口径。码本结构参数（L、M_v、β 等）取自 [3] 原文，不属于性能数值。
- [6] 表 I 的 GCS 数值未写入正文，只作为"二手转引"记录在核查记录中；本文不据此做任何比较性判断。
- 未记录任何参数量或 FLOPs：所检索文献的摘要均未给出可与本仓库对齐的参数量/FLOPs 口径。

**检索词与日期**

检索日期 2026-11-03。主要检索词：`3GPP TR 38.843`、`Type II codebook overhead`、`eType II frequency domain compression`、`FeCoMP CSI`、`CSI compression coherent joint transmission`、`two-sided CSI compression standardization`、`AI native air interface 3GPP`、`joint pilot design CSI feedback`、`precoding-oriented CSI feedback`、`CSI codebook deep learning comparison`、`EVCsiNet`、`AI beam management 3GPP Rel-18`。

**局限**

- 3GPP 贡献文档（R1-xxxxxxx）未逐篇检索，码本参数的原始提案依据来自综述 [3] 与规范 [2]，因此"某参数由哪家公司提出"一类问题不在本文覆盖范围。
- 2026 年的多条文献为预印本，尚未经过同行评审；本文只把它们作为"最新动向"记录，未据此下结论。
- 本文未讨论 Rel-19/20 的具体条款编号，相关表述以综述 [8][9] 为准，细节仍需以 3GPP 正式文本复核。
