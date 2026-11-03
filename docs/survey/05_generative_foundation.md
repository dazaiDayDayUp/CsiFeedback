# 生成式模型、自监督预训练与基础模型路线

> 本文只做文献梳理，不提出新方法。数值均标注来源层级（原文全文 / 摘要 / 二手转述）；
> 各论文训练预算、量化方式与评测协议差异很大，**不做跨论文的排名式断言**。

## 1. 范围与检索方法

本文关注三条与 CSI 反馈重建端直接相关的路线：**显式分布建模**（GAN/VAE，把重建视为条件分布采样）、
**扩散 / score-based 生成**（迭代去噪实现后验采样，近三年从信道估计迁移到 CSI 压缩解码）、
**自监督预训练与基础模型**（掩码重建、对比学习、多配置统一预训练、LLM 试探）。

主干是下行 CSI 压缩与重建。信道生成与信道估计被纳入，因为它们是生成式方法进入物理层的方法来源，
其评测思路（分布距离、导频开销）也直接影响 CSI 反馈论文的公平性判断。

检索于 2026-11-03 完成，使用 arXiv API 的标题/摘要检索、网页检索，以及 Semantic Scholar / Crossref 元数据核对。
主要检索词：`CSI feedback diffusion`、`generative adversarial network CSI feedback`、
`variational autoencoder CSI compression`、`score-based channel estimation`、`wireless foundation model CSI`、
`masked pretraining CSI`、`large language model CSI feedback`。

## 2. 时间线与分类

| 年份 | 代表工作 | 路线标记 |
|------|----------|----------|
| 2018 | CsiNet 确立「自编码器 + 量化码字」范式 | 判别式回归（参照系） |
| 2020 | Tolba et al. GAN 反馈；PRVNet（VAE） | 分布建模 |
| 2022 | ChannelGAN 用于信道建模与生成 | GAN 生成信道 |
| 2023 | score-based MIMO 信道估计；扩散信道采样 | 条件采样 |
| 2024 | LWM 自监督信道基础模型；多速率 VAE + 熵瓶颈 | 预训练 / 可变码率 |
| 2025 | 扩散压缩（含边信息）、RD-JSCC、WiFo-CF、LLM4CSI、CSI-GPT | 生成解码 + 基础模型并行 |
| 2026 | WiFo-2、HeterCSI、CsiCoGen、PAC、多模态物理层基础模型 | 统一预训练 + 任意码长生成 |

演进可概括为：**研究重心从「更好的回归器」转向「更好的先验」**。
2018–2021 年竞争点是编码器结构与量化损失；2023 年后学界承认低压缩比重建是欠定问题，
改进点转到解码端用什么分布补足信息；2025–2026 年则把「先验从哪来」外化为预训练规模与异构配置覆盖。

## 3. 代表性工作

### 3.1 GAN 与 VAE：把重建写成条件分布

**Tolba et al., "Massive MIMO CSI Feedback Based on Generative Adversarial Network", IEEE Communications Letters, 2020.** [1]
据二手描述，该工作在自编码器重建端引入对抗训练：生成器（解码器）从压缩向量恢复 CSI，判别器区分真伪；
动机是 MSE 训练倾向输出条件均值、低码率下偏平滑。原文全文未获取，结构与数值不作断言。

**Hussien, Nguyen, Cheriet, "PRVNet: A Novel Partially-Regularized Variational Autoencoders for Massive MIMO CSI Feedback", arXiv:2011.04178, 2020.** [2]
把 CSI 压缩写成 VAE：编码器输出隐变量后验，解码器重建，KL 项约束隐空间。
作者强调 CSI 定制损失与 KL-annealing 调节正则，说明核心难点是**先验强度与保真度的权衡**。
论文同时考察 AWGN 反馈信道下的鲁棒性。venue 未核实。

**Park, Do, Lee, "Multi-Rate Variable-Length CSI Compression for FDD Massive MIMO", ICASSP 2024 (arXiv:2311.18172).** [3]
用 VAE + entropy bottleneck 实现多速率、变长码字：单模型覆盖多码率，码字长度随样本自适应。
摘要只给「NMSE 优于现有方法」的定性结论，无具体 dB 值。

**Liu, Cao, Ohtsuki, He, Mumtaz, "Generative Model-Aided Continual Learning for CSI Feedback in FDD mMIMO-OFDM Systems", arXiv:2511.19490, 2025.** [4]
把 GAN 生成器当作**记忆单元**：持续学习新环境时重放旧环境 CSI 分布，缓解灾难性遗忘，
维持低内存开销下的跨场景性能。这是生成能力服务于「不遗忘」而非「补细节」的少见用法。

**Xiao et al., "ChannelGAN: Deep Learning-Based Channel Modeling and Generating", IEEE Wireless Communications Letters, 11(3):650–654, 2022.** [22]
用 GAN 学习信道分布并生成信道实现，提供了「生成质量在分布层面评价」的思路。

### 3.2 扩散与 score-based：条件后验采样

**Arvinte & Tamir, "MIMO Channel Estimation Using Score-Based Generative Models", IEEE Transactions on Wireless Communications, 22(6):3698–3713, 2023.** [5]
训练 score network 估计对数分布梯度，测试时用后验采样迭代修正。论文给出 SISO 下的鲁棒性理论保证，
并报告：分布外性能明显优于监督学习，端到端编码通信增益最高 5 dB；25% 导频密度下可对最高 64×256 的
MIMO 信道做高保真估计；FLOPs 与压缩感知相当。

**Sengupta, Jao, Bernacchia, Vakili, Shiu, "Generative Diffusion Models for Radio Wireless Channel Modelling and Sampling", IEEE GLOBECOM 2023.** [6]
在频域-空域上用 U-Net 扩散合成信道，以 2-Wasserstein 距离（归一化功率谱）与 precision/recall 评价分布。
论文明确指出 GAN 基线存在 mode collapse 与训练不稳定，而扩散训练稳定、样本多样；
并演示了宏蜂窝预训练 + 少量微蜂窝微调的可行性。

**Kim, Lee, Kim, De Veciana, Arfaoui, Koc, Pietraski, Zhang, Kaewell, "Generative Diffusion Model-Based Compression of MIMO CSI", IEEE ICC 2025.** [7]
解码端写成以「码字 + 边信息」为条件的反向扩散：编码端用可训练码本做定长 VQ，解码端用 DDIM 采样。
任务是「用当前 DL CSI 预测未来 DL CSI」，边信息为 UL CSI，另给纯重建结果。
COST2100 outdoor（10^5 训练 / 2×10^4 测试）上，约 −7 dB 失真需少于 150 bit，基线需 400 bit 以上；
T=4 步，训练 3×10^5 步、batch 100、lr 1e-3（原文全文）。**其 COST2100 基线曲线直接引自
CsiConformer、CsiNet+、深度自编码器熵编码三篇论文，未在同一实验室内重跑。**

**Ankireddy, Kim, Cho, Kim, "Residual Diffusion Models for Variable-Rate Joint Source Channel Coding of MIMO CSI", IEEE JSAC, 2026.** [8]
针对分离信源-信道编码的悬崖效应，用轻量 AE 给出粗估计，再由**残差扩散**从粗估计初始化并精修，
而非从纯高斯噪声开始。解码可在「仅 AE」与「AE + 扩散」间按信道条件切换，支持单模型多码率、
固定比特量化与不完美信道估计；推理扩散仅需 2 步（摘要）。

**Cheng, Meng, Zou, Xie, Zhang, Wang, Liang, "Variable-Length Finite-Rate CSI Feedback With Generative Priors"（CsiCoGen）, IEEE Wireless Communications Letters, 已接收, 2026.** [9]
把有限比特决策移到反向扩散轨迹上的「码本约束高斯更新选择」：收发共享免训练伪随机码本，
每个索引对应一次生成更新，长度 L 的前缀即产生一个有效估计，码率 R=L·log2K。
COST2100 上报告 792 bit 时 indoor/outdoor NMSE −28.58/−13.96 dB、rho 0.9964/0.9597；
1592 bit 时 −30.72/−20.37 dB、rho 0.9967/0.9748（摘要）。其指标以比特数报告，与本仓库 CR 网格不能直接对齐。

**Yao, You, Zhou, Gong, Xing, "Generative Deep Learning for CSI Feedback in Massive MIMO via Channel Distribution Prediction"（PAC）, arXiv:2610.16796, 2026.** [10]
两阶段：先用因果 Transformer 预测下一帧 CSI token 的条件分布，帧到达后用该分布做变长熵编码。
token 不依赖历史 CSI，故对突然变速稳健。作者报告相对历史感知基线降低 72%–84% 开销（摘要）。
其生成物是**概率模型而非样本**，与扩散解码形成对照。

### 3.3 自监督与掩码预训练

**Alikhani, Charan, Alkhateeb, "Large Wireless Model (LWM): A Foundation Model for Wireless Channels", arXiv:2411.08872, 2024.** [11]
Transformer 在大规模信道数据上自监督预训练，输出通用信道嵌入供下游任务使用。
作者报告以 LWM 嵌入替代原始信道表示后，高复杂度与小样本下游任务性能一致提升（摘要）。
它确立了「信道数据可做自监督预训练」这一前提。

**Liu, Gao, Liu, Cheng, Yang, "WiFo-CF: Wireless Foundation Model for CSI Feedback", arXiv:2508.04068, 2025.** [12]
首个明确面向 CSI 反馈的基础模型：多用户多速率自监督预训练 + 「共享-路由专家」MoE，
统一接纳不同信道维度、反馈速率与数据分布，并配套异构反馈数据集。
作者报告分布内/外、仿真/实测均有优势，表征可迁移至室内定位（摘要）。

**Zhang, Lyu, Ren, Liu, Cui, Tao, "HeterCSI: Channel-Adaptive Heterogeneous CSI Pretraining Framework for Generalized Wireless Foundation Models", arXiv:2601.18200, 2026.** [14]
针对 CSI 的尺度异构与场景异构：把异构 batch 构造形式化为分区优化以减少 zero-padding，
并用 MAE 式双重掩码分离有效信号与填充伪影。12 个数据集上无需场景微调即泛化，
相对 WiFo 零样本基线在 CSI 重建上降低 NMSE 7.19 dB，训练时延降低 53%（摘要）。
其论点是「尺度异构造成梯度破坏性干扰，场景多样性反而带来建设性对齐」。

**Ryu, Kim, Lee, Yang, Cioffi, "Self-Supervised Pretraining for Transmission-Relevant MIMO CSI Representation", arXiv:2607.24872, 2026.** [15]
提出 beam-response contrastive learning：用发射侧 Gram 矩阵 R=H^H H 诱导的波束响应相似度
替代通用数据增强构造正负对。作者报告其优于自编码器式与 channel-charting 式预训练（摘要）。
它指出 CSI 自监督的关键问题：**「相似」应由传输性能定义，而非像素级重构**。

**Song, Yang, Zhi, Li, Wu, Xue, Caire, "Prior-Aided Masked Vector Quantization CSI Feedback for FDD Massive MIMO Systems", arXiv:2609.17769, 2026.** [16]
非扩散，但机制属「先验补全」：以平均角度-时延功率图为条件，掩码按反馈预算挑选 token，
量化后连同位置反馈，接收端 de-masking 补全；单模型覆盖多个 token 数。
在 Sionna 生成的 3GPP TR 38.901 UMa 数据上，作者报告弥散信道区域所有测试 SNR 与反馈维度下 NMSE 最低（摘要），
即低码率歧义可依靠二阶统计先验消解，开销远低于生成采样。

### 3.4 基础模型与 LLM

**Liu, Liu, Gao, Cai, Cheng, Yang, "WiFo-2: A Generalist Foundation Model Unifies Heterogeneous Wireless System Design", arXiv:2511.22222, 2025.** [13]
在 116 亿 CSI 点上预训练的空-时-频基础模型，作者报告其呈现 scaling-law 行为，
零样本信道重建优于全监督专用模型，仅用 1% 样本即在 9 类任务上达 SOTA，并有硬件原型（摘要）。

**Cui, Guo, Wen, Jin, Tong, "Exploring the Potential of Large Language Models for Massive MIMO CSI Feedback", IEEE GLOBECOM 2025.** [17]
把 LLM 当「去噪器」：为 CSI 设计预处理、嵌入与后处理，将压缩 CSI 映射进 token 空间，利用纠错能力改善重建。
摘要只给「有前景」的定性结论，无 dB 数值。

**Zeng, Qiao, Gao, Qin, Wu, Khalaf, Chen, Guizani, "CSI-GPT: Integrating Generative Pre-Trained Transformer With Federated-Tuning to Acquire Downlink Massive MIMO Channels", IEEE Transactions on Vehicular Technology, 74(3):5187–5192, 2025.** [18]
GPT + 联邦微调，配 Swin Transformer 信道获取网络，覆盖导频、估计与反馈整条链路。
出发点是降低每基站的重训成本，而非单点 NMSE 提升。

**Yazdnian & Ghasempour, "A Multi-Modal Foundational Model for Wireless Communication and Sensing", arXiv:2602.04016, 2026.** [19]
任务无关的多模态物理层基础模型，以「物理 token」在自监督预训练中捕捉跨模态电磁传播对应关系，
再以少量标注适配多天线优化、信道估计与定位。作者报告在泛化性、部署漂移鲁棒性与数据需求上优于任务专用基线（摘要）。

**Cheng, Liu, Liu, Cai, "Large Wireless Foundation Models: Stronger over Bigger", arXiv:2601.10963, 2026.** [20]
立场/综述文章：讨论复用通用基础模型与自建无线基础模型两条路线，给出路线图与无线约束下的设计原则，
并从多维界定「大」的含义。不提供新实验结果。

**Liang, Guo, Zhang, Chae, Lu, Xu, Dobre, Jin, Li, "Foundation Models for Wireless Communications: From PHY Intelligence to Network Autonomy", arXiv:2606.06239, 2026.** [21]
综述，把基础模型在无线的应用分为「适配现成模型」「无线原生模型」「智能体化模型」三层，并讨论 ISAC 与新 MIMO 架构。

## 4. 三个横切问题

### 4.1 低压缩比下的重建歧义

文献的处理可归为四类，代价依次上升：

1. **只改损失**（GAN 判别器、VAE 的 KL 项）[1][2]：推断仍是单次前向，但无法控制生成多样性，
   且对抗训练本身不稳定（Sengupta 等明确指出 GAN 的 mode collapse）[6]。
2. **二阶统计先验补全**[16]：先验收发共享、无迭代采样，但表达力受限于功率图这类低维统计量。
3. **条件后验采样**[7][8][9]：最贴合欠定逆问题的表述，但需多步前向且单次采样带随机性。
4. **把歧义转成概率模型**[10]：预测 token 分布再熵编码，折中表达力与延迟。

需要强调：**这四类没有在同一协议下被比较过**。第 3 类的三篇分别用「未来帧预测 + 边信息」
「JSCC + 残差」「任意码长前缀」三种任务设定，指标口径（失真 dB / 比特数 / 开销百分比）也不一致。

### 4.2 采样与推理开销

三种降本手段在文献中可见：

- **减步数**：Kim 等 T=4 的 DDIM [7]；RD-JSCC 2 步，并允许信道好时完全跳过扩散 [8]。
- **改初始化**：RD-JSCC 从 AE 粗估计出发做残差扩散 [8]，同时降低步数与样本方差。
- **改生成对象**：CsiCoGen 让每个码本索引对应一次生成更新，长度 L 的前缀直接给出估计，
  避免「先收全码字再采样」的延迟，并报告加速采样吞吐 [9]。

但**只有少数论文报告时延、FLOPs 或吞吐**。GAN/VAE 单次前向在部署侧更有优势，
生成式的精度增益往往以解码端算力换取。

### 4.3 评测公平性

本调研在所读材料中观察到四个具体问题：

1. **基线未重跑**。Kim 等 [7] 的 COST2100 基线曲线直接引自三篇论文，各基线量化方式
   （6-bit 均匀量化、CsiNet+ 的 offset network、熵编码的可变长码字）不同，
   比较混合了架构差异与量化/编码差异。
2. **码率表述不统一**。分别用比特数 [7][9]、压缩比、或相对基线的开销百分比 [10] 报告；
   CsiCoGen 的 792/1592 bit 映射到 CR 网格需假设每维比特数，原文未给出（未核实）。
3. **训练预算差异大且常不报告**。Kim 等给出 3×10^5 步 [7]；基础模型论文多只报告数据规模与任务数 [12][13][14]，
   难以判断增益来自先验还是算力。
4. **生成随机性未进入指标**。所读材料中 NMSE/rho 均按单次或固定种子评估，
   未见多次采样的均值 ± 方差；对扩散解码器，同一码字可能对应质量差异明显的多个重建。

此外，**checkpoint 选择策略**普遍未说明。本仓库 baseline 使用 `select_best: "all"`（测试集指标选模型），
若与用验证集选模型的新方法对比，必须显式注明。

## 5. 对比表

下表只列任务设定可比性较高的四篇生成式 CSI 反馈工作。空缺表示原文（摘要或所读正文）未提供。

| 工作 | 压缩接口 | 解码机制 | 报告数值（原文口径） | 采样开销 |
|------|----------|----------|----------------------|----------|
| Kim et al., ICC 2025 [7] | 可训练码本 + 定长 VQ | 码字与边信息条件下的 DDIM | COST2100 outdoor：约 −7 dB 需 <150 bit，基线 >400 bit（全文） | T=4 步 |
| RD-JSCC, JSAC 2026 [8] | 轻量 AE + 可变码率 | 粗估计初始化的残差扩散，可与纯 AE 切换 | 摘要仅给「显著优于 AE 基线」，无 dB 值 | 2 步，可跳过 |
| CsiCoGen, WCL 2026 [9] | 码本索引前缀，R=L·log2K | 码本约束高斯更新沿反向轨迹 | 792 bit：−28.58/−13.96 dB（in/out），rho 0.9964/0.9597（摘要） | 报告加速采样吞吐，步数未在摘要给出 |
| PAC, 2026 [10] | token 分布 + 变长熵编码 | 因果 Transformer 预测分布 + 并行因果解码 | 相对历史感知基线降低 72%–84% 开销（摘要） | 非扩散，无迭代采样 |

**读表注意**：Kim 与 CsiCoGen 不可直接比较——前者是未来帧预测、失真在裁剪后的角度-时延域计算，
后者是单帧重建；比特数含义也不同（定长码字 vs 码本索引前缀）。

## 6. 小结

- GAN/VAE 最早把「重建 = 条件分布」引入 CSI 反馈：GAN 提供感知约束但训练不稳，
  VAE 提供显式隐空间与码率控制（熵瓶颈），代价是重建偏平滑。两者多作为自编码器的附加模块。
- 扩散/score-based 路线 2023 年先落地于信道估计与信道生成，2025 年进入 CSI 压缩解码。
  三篇代表工作分别用边信息条件化、残差初始化、码本前缀降低采样成本，共同点是
  **把不确定性显式建模**而非追求单步确定性映射。
- 自监督与基础模型解决的是「跨配置复用」而非「低码率重建」。WiFo-CF 与 HeterCSI 都把码率异构当作核心问题，
  与本仓库覆盖 1/4~1/64 的设定直接相关；LLM 路线仍停留在可行性验证。

## 7. 文献明确指出的开放问题

以下均为原作者在摘要或正文中自己提出的问题：

- 扩散解码的计算复杂度仍是部署障碍，需要更轻量结构与更快采样（Kim 等，第 V 节）[7]。
- 分离信源-信道编码存在悬崖效应；AE 类 JSCC 难刻画复杂信道分布且参数量扩展性差（RD-JSCC）[8]。
- 现有可伸缩方案把有限比特接口与任务特定 latent codec 耦合，换码率就要换编码器（CsiCoGen）[9]。
- CSI 尺度异构导致梯度破坏性干扰，zero-padding 造成训练浪费（HeterCSI）[14]。
- 现有 CSI 反馈模型受固定系统配置约束，跨维度/速率/分布泛化不足（WiFo-CF）[12]。
- 无线约束使 LLM 式「规模换能力」不能直接照搬，需明确 LWFM 设计原则（LWFM）[20]。
- 通用数据增强构造的对比学习正负对不反映传输相关性（BRCL）[15]。

## 8. 参考文献

[1] B. Tolba, M. Elsabrouty, M. G. Abdu-Aguye, H. Gacanin, H. M. Kasem. Massive MIMO CSI Feedback Based on Generative Adversarial Network. IEEE Communications Letters, 24(12):2805–2808, 2020. DOI: 10.1109/LCOMM.2020.3017188.

[2] M. Hussien, K. K. Nguyen, M. Cheriet. PRVNet: A Novel Partially-Regularized Variational Autoencoders for Massive MIMO CSI Feedback. arXiv:2011.04178, 2020. https://arxiv.org/abs/2011.04178

[3] B. Park, H. Do, N. Lee. Multi-Rate Variable-Length CSI Compression for FDD Massive MIMO. IEEE ICASSP 2024 (arXiv:2311.18172). https://arxiv.org/abs/2311.18172

[4] G. Liu, Y. Cao, T. Ohtsuki, J. He, S. Mumtaz. Generative Model-Aided Continual Learning for CSI Feedback in FDD mMIMO-OFDM Systems. arXiv:2511.19490, 2025. https://arxiv.org/abs/2511.19490

[5] M. Arvinte, J. I. Tamir. MIMO Channel Estimation Using Score-Based Generative Models. IEEE Transactions on Wireless Communications, 22(6):3698–3713, 2023. https://arxiv.org/abs/2204.07122

[6] U. Sengupta, C. Jao, A. Bernacchia, S. Vakili, D. Shiu. Generative Diffusion Models for Radio Wireless Channel Modelling and Sampling. IEEE GLOBECOM 2023. https://arxiv.org/abs/2308.05583

[7] H. Kim, T. Lee, H. Kim, G. De Veciana, M. A. Arfaoui, A. Koc, P. Pietraski, G. Zhang, J. Kaewell. Generative Diffusion Model-Based Compression of MIMO CSI. IEEE ICC 2025. https://arxiv.org/abs/2503.03753

[8] S. K. Ankireddy, H. Kim, J. Cho, H. Kim. Residual Diffusion Models for Variable-Rate Joint Source Channel Coding of MIMO CSI. IEEE Journal on Selected Areas in Communications, 2026 (to appear). https://arxiv.org/abs/2505.21681

[9] Y. Cheng, F. Meng, J. Zou, J. Xie, Z. Zhang, Y. Wang, Y. Liang. Variable-Length Finite-Rate CSI Feedback With Generative Priors. IEEE Wireless Communications Letters, accepted, 2026. https://arxiv.org/abs/2606.06846

[10] L. Yao, C. You, C. Zhou, Y. Gong, C. Xing. Generative Deep Learning for CSI Feedback in Massive MIMO via Channel Distribution Prediction. arXiv:2610.16796, 2026. https://arxiv.org/abs/2610.16796

[11] S. Alikhani, G. Charan, A. Alkhateeb. Large Wireless Model (LWM): A Foundation Model for Wireless Channels. arXiv:2411.08872, 2024. https://arxiv.org/abs/2411.08872

[12] X. Liu, S. Gao, B. Liu, X. Cheng, L. Yang. WiFo-CF: Wireless Foundation Model for CSI Feedback. arXiv:2508.04068, 2025. https://arxiv.org/abs/2508.04068

[13] B. Liu, X. Liu, S. Gao, X. Cai, X. Cheng, L. Yang. WiFo-2: A Generalist Foundation Model Unifies Heterogeneous Wireless System Design. arXiv:2511.22222, 2025. https://arxiv.org/abs/2511.22222

[14] C. Zhang, X. Lyu, C. Ren, S. Liu, Q. Cui, X. Tao. HeterCSI: Channel-Adaptive Heterogeneous CSI Pretraining Framework for Generalized Wireless Foundation Models. arXiv:2601.18200, 2026. https://arxiv.org/abs/2601.18200

[15] S. Ryu, Y. Kim, M. Lee, H. J. Yang, J. M. Cioffi. Self-Supervised Pretraining for Transmission-Relevant MIMO CSI Representation. arXiv:2607.24872, 2026. https://arxiv.org/abs/2607.24872

[16] Y. Song, T. Yang, K. Zhi, S. Li, F. Wu, S. Xue, G. Caire. Prior-Aided Masked Vector Quantization CSI Feedback for FDD Massive MIMO Systems. arXiv:2609.17769, 2026. https://arxiv.org/abs/2609.17769

[17] Y. Cui, J. Guo, C.-K. Wen, S. Jin, E. Tong. Exploring the Potential of Large Language Models for Massive MIMO CSI Feedback. IEEE GLOBECOM 2025. https://arxiv.org/abs/2501.10630

[18] Y. Zeng, L. Qiao, Z. Gao, T. Qin, Z. Wu, E. Khalaf, S. Chen, M. Guizani. CSI-GPT: Integrating Generative Pre-Trained Transformer With Federated-Tuning to Acquire Downlink Massive MIMO Channels. IEEE Transactions on Vehicular Technology, 74(3):5187–5192, 2025. DOI: 10.1109/TVT.2024.3493463.

[19] V. Yazdnian, Y. Ghasempour. A Multi-Modal Foundational Model for Wireless Communication and Sensing. arXiv:2602.04016, 2026. https://arxiv.org/abs/2602.04016

[20] X. Cheng, B. Liu, X. Liu, X. Cai. Large Wireless Foundation Models: Stronger over Bigger. arXiv:2601.10963, 2026. https://arxiv.org/abs/2601.10963

[21] L. Liang, J. Guo, J. Zhang, C.-B. Chae, L. Lu, S. Xu, O. A. Dobre, S. Jin, G. Y. Li. Foundation Models for Wireless Communications: From PHY Intelligence to Network Autonomy. arXiv:2606.06239, 2026. https://arxiv.org/abs/2606.06239

[22] H. Xiao et al. ChannelGAN: Deep Learning-Based Channel Modeling and Generating. IEEE Wireless Communications Letters, 11(3):650–654, 2022. DOI: 10.1109/LWC.2021.3140102.

## 9. 核查记录

**检索日期**：2026-11-03。**工具**：arXiv API（按 id_list 批量取标题/作者/摘要/日期）、网页检索、
Semantic Scholar 与 Crossref 元数据。

**信息层级**：

- **原文全文（A 级）**：仅 Kim et al. [7]。通读 arXiv HTML 全文；T=4、3×10^5 步、batch 100、lr 1e-3、
  COST2100 outdoor 10^5/2×10^4 划分、基线曲线引自他人论文、−7 dB 对应 <150 bit vs >400 bit 均取自正文。
- **摘要（B 级）**：[2]–[6]、[8]–[22]。网络细节、超参数与表格数值未逐篇核对全文，故未写作事实。
- **二手来源（C 级）**：
  - Tolba et al. [1]：IEEE Xplore 与 DOI 页面需 JavaScript，无法抓取。「生成器重建、判别器判别真伪」
    来自综述/行业刊物转述，**未获原文证实**，正文已标注。
  - ChannelGAN [22]：标题、卷期、页码与 DOI 来自检索结果引文字段，摘要未获取；正文只描述其定位。
  - PRVNet [2] 的 venue：未核实，按 arXiv 引用。
- **元数据核对**：[17] 的 GLOBECOM 2025 与 DOI 经 Semantic Scholar 确认；[18] 卷期页码与 DOI
  经出版方与机构库交叉确认；[9] 的「WCL 已接收」取自 arXiv comment 字段。

**已知不确定项**：

1. CsiCoGen 的 792/1592 bit 与 CR 1/4~1/64 的对应关系原文未给出，本文未换算。
2. Kim et al. 的数值来自未来帧预测、失真在裁剪后的角度-时延域计算，与单帧重建不同口径。
3. 「多数论文未报告采样方差 / checkpoint 选择」是基于所读材料的观察，不构成统计结论。
4. [10]、[14]、[15]、[16]、[19]、[20]、[21] 为近月预印本，结论可能随版本变化。

**未覆盖**：3GPP 码本与 AI/ML 空口标准化（属 `07_standards_adjacent.md`）、
CSI 反馈的隐私与安全、生成式模型在 CSI 压缩中的理论率失真界。后两者在本主题下文献较少，未强行凑数。
