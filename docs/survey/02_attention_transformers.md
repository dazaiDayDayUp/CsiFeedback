# 注意力与 Transformer 类 CSI 反馈方法（2019–2026）

## 1. 范围与检索方法

本文聚焦 FDD 下行 CSI 的压缩反馈：UE 将角-时延域截断后的 `(2, 32, 32)` CSI 编码为长度 `2048/CR` 的码字，BS 解码重建；压缩率 CR 取 1/4–1/64，评测以 COST2100 indoor/outdoor 的 NMSE（dB）与 rho 为主。CsiNet [1] 确立了这一自编码器范式，本仓库的 CLNet、STNet 也在同一协议下评测。

纳入标准是注意力或 Transformer 位于压缩/重建主路径，且能说明其作用域（空间、通道、窗口、全局）。三类相邻工作单独标注、不与 COST2100 结果并列：图像压缩感知的 CSformer [18]、联合导频/估计/反馈的 CSI-GPT [17]、近场 XL-MIMO 的 NEFT [12]。

检索于 2026-11-03 进行，来源为 arXiv API 与论文页、Google Scholar、Semantic Scholar Graph API、Crossref、IEEE Xplore 摘要页与作者 GitHub。检索词包括 "CSI feedback transformer"、"attention CSI feedback"、"spatially separable attention"、"Swin CSI feedback"、"CSI feedback XL-MIMO transformer" 等。

命名核查有两点。其一，按 "DS-Net" 在 arXiv 与 Google Scholar 检索，未找到以该名发表的 CSI 反馈工作；最接近的是 DS-NLCsiNet [3]（非局部网络），下文按后者处理。其二，CSFormer 与 CSIFormer 是两篇不同论文：CSformer [18] 是图像压缩感知方法（Ye et al., TIP 2023），CSIFormer [8] 是 CSI 反馈的卷积-窗口 Transformer（Bi et al., WCL 2022）。STNet 的参考文献同时引用二者，名称极易混淆。

## 2. 时间线与分类

- 2019–2021：注意力作为卷积自编码器的增强模块，在 CsiNet 特征上做非局部或自注意力加权。
- 2021–2022：把 Transformer 编码器/解码器直接用于 CSI，出现全注意力自编码器。
- 2022–2024：为 CSI 的角-时延二维结构设计窗口、可分离、移位窗口注意力，重点转向复杂度。
- 2024–2026：层级化与端侧非对称设计，并把注意力与量化、熵编码、生成模型结合。

按作用域可分为五类：A 特征重标定与长程建模 [2][3][4]；B 全注意力自编码器 [5][6][7]；C 窗口/可分离/移位窗口 [8][9][10][11]；D 层级化与端侧非对称 [12][13]；E 跨流与混合编码 [14][15][16]。CSI-GPT [17] 属联合获取，CSformer [18] 属方法来源。

四条作用域对应不同的计算结构。设特征图为 `L×L`（本文 L=32），全局自注意力的 token 数为 `L^2`，注意力矩阵为 `L^2×L^2`，STNet 将其记为 `O(L^4 d)`；窗口注意力把每个窗口的矩阵降到 `W^2×W^2`，代价是窗口之间不通信；下采样 KV 的全局注意力用 `m^2` 个汇总 token 近似全局，复杂度为 `O(m^2 L^2 d)`；通道注意力在通道维做重标定，计算量随通道数线性增长，但不直接建模天线间的空间相关；移位窗口则在相邻层交换窗口边界信息，避免固定分区的隔离。上述取舍在 [9][10] 中有明确的复杂度与实现描述。

区分“真正做注意力”与“把 Transformer 模块拼进自编码器”，本调研采用三条可核查标准：(i) token 如何定义，是否对应 CSI 的角-时延结构；(ii) 是否针对 CSI 的局部/长程相关性改造注意力（窗口、下采样 KV、跨流交换等）；(iii) 是否报告复杂度并与全注意力对比。三条均满足者视为注意力设计；仅把标准 Transformer block 接在卷积特征之后、token 仍是展平特征且无复杂度分析的，视为模块拼接。按此标准，[6][9][10] 三条均满足；[8] 的窗口设计满足前两条，复杂度对比由 STNet 转述而非原文核实。部分未单列的“ViT + 自编码器”工作属模块拼接（本调研判断）。

## 3. 代表性工作

### 3.1 注意力作为卷积特征的增强（2019–2021）

**Attention-CsiNet** [2]（Cai, Dong, Niu，WCNC 2019）在 CsiNet 的特征上引入注意力模块，对特征进行加权后再解码。据 STNet 的转述，其增益在相关性变化更剧烈的 outdoor 更明显 [9]。SwinCFNet 的汇总表列出其 COST2100 值为 indoor −20.29 dB、outdoor −10.43 dB（CR=1/4），但该表为二手汇总，原始训练预算未核实 [10]。

**DS-NLCsiNet** [3]（Yu et al.，IEEE Communications Letters 2020）用非局部块（non-local block）捕获 CSI 的长程依赖，并以密集连接强化特征细化模块；作者报告在相同压缩率下重建精度优于当时的压缩方案。它是“注意力作为长程建模插件”的代表，未改变 CsiNet 的整体自编码器骨架。

**SALDR** [4]（Song et al.，IEEE Wireless Communications Letters 2021）把联合自注意力学习与密集细化（dense refine）结合，面向多压缩率场景。可访问的摘要片段只说明其以自注意力提升重建精度，具体结构与数值未能核实，故不引用其结果。

### 3.2 全注意力 Transformer 自编码器（2021–2022）

**CSITransformer** [5]（Xu, Yuan, Pun，WOCC 2021）较早把 Transformer 层用于 CSI 反馈，并讨论了自注意力层的计算复杂度。STNet 评述其性能不及同期 SOTA，且实验数据集与 COST2100 不同，因此后续表格未并列其 NMSE [9]。

**TransNet** [6]（Cui, Guo, Song，IEEE Wireless Communications Letters 2022）是两层 Transformer 自编码器，编码器与解码器各两层。TransNet+ 在描述其继承的结构时指出，它以角-时延矩阵的行向量为 token，编码器与解码器均使用多头自注意力，最后由线性层输出重建 [7]。作者仓库给出 COST2100、32×32、1000 epoch、batch 200、常数学习率 1e-4 的完整结果：indoor NMSE 为 −32.38/−22.91/−15.00/−10.49 dB，outdoor 为 −14.86/−9.99/−7.82/−4.13 dB（CR 1/4→1/32）；FLOPs 在 CR=1/4 与 1/64 分别为 35.72M 与 33.75M。同一模型 400 epoch 时 indoor 1/4 仅 −29.22 dB，与 1000 epoch 相差 3.16 dB，说明训练预算对报道值影响显著。

**TransNet+** [7]（Chen, Guo, Cui，Applied Sciences 2024）在 TransNet 上把特征维度从 32 增至 64、注意力头数从 2 增至 4，并改用 warm-up cosine 学习率。作者报告相对 TransNet 的 indoor 增益为 0.74/0.56/0.70/1.49/1.91 dB（CR 1/4→1/64），高压缩率处增益更大；同时明确指出未考虑码字量化，实际部署仍需补足。

### 3.3 面向 CSI 结构的局部/窗口注意力（2022–2026）

**CSIFormer** [8]（Bi, Li, Yu, Zhang，IEEE Wireless Communications Letters 2022）采用卷积-Transformer 混合结构，以窗口自注意力建模长程依赖，重点是高压缩率下的性能。摘要报告 CR=1/64 时平均精度较当时 SOTA 提升 5.39%。STNet 将其描述为局部窗口自注意力，复杂度低于全注意力但性能有所牺牲 [9]。

**STNet** [9]（Mourya, Amuru, Kuchi，arXiv 2022；IEEE Wireless Communications Letters 2023，vol. 12, no. 1；本仓库 baseline）提出空间可分离注意力：LSA 在 8×8 窗口内做多头自注意力，GSA 用 stride=8 的卷积把窗口特征下采样为 K/V、以 LSA 输出为 Q 做跨窗口全局注意力。文中给出复杂度：全注意力为 `O(L^4 d)`，LSA 为 `O(L^4 d / m^4)`，GSA 为 `O(m^2 L^2 d)`。COST2100 indoor、CR=1/4 时总 FLOPs 为 5.22M，其中编码器 2.09M（40.03%），CLNet 为 4.42M；CR=1/16 与 1/64 时分别只需 TransNet 11.6% 与 10.8% 的 FLOPs。解码器采用 CNN 与 Transformer 双 stem。

**SwinCFNet** [10]（Cheng et al.，arXiv 2024）用两阶段 Swin Transformer 自编码器：2×2 patch、W-MSA 与 SW-MSA 交替、`[N1,N2]=[2,4]`、heads `[4,8]`，训练 1000 epoch、batch 200、常数学习率 1e-4。其 Table I 报告 indoor NMSE 为 −28.94/−21.09/−15.12/−9.39 dB，outdoor 为 −21.29/−15.90/−11.50/−7.62 dB（CR 1/4→1/32）。与 TransNet 相比，indoor 1/4 高 3.44 dB（更差），outdoor 1/4 低 6.43 dB（更好）；作者把 outdoor 的收益归因于移位窗口的层级特征更适合分散的非零结构。

**SwinCANet** [11]（Tao et al.，arXiv 2026）先用 curvelet 变换把 CSI 分为低频/高频分量，低频用 Swin Transformer、高频用通道注意力分别建模，再用额外的 Swin Transformer 融合多尺度分量；变体 De-SwinCANet 用 sigmoid 阈值抑制噪声系数。摘要只给出“优于现有基准、在恶劣传播条件下更稳健”的定性结论，未给出可核实的 COST2100 数值。

### 3.4 层级化与端侧非对称（2025）

**NEFT** [12]（Mao et al.，arXiv 2025）面向近场 XL-MIMO，采用层级视觉 Transformer 主干、渐进 token 缩减与多尺度特征；NEFT-Compact 用多级知识蒸馏压缩模型，NEFT-Hybrid 改用无注意力的 CNN 编码器以降低 UE 侧计算（编码端复杂度最多降 64%），NEFT-Edge 面向极受限设备。摘要报告 NMSE 较 SOTA 改善 15–21 dB，但这是近场 XL-MIMO 基准，不是 COST2100 32×32 协议，不能与本仓库结果并列。

**AiANet** [13]（Lou, Wu，IEEE Transactions on Wireless Communications 2025）用通道注意力与空间注意力双分支并行提取特征并融合，另设局部感知自注意力（LASA）同时捕捉全局与局部空间模式；混合训练（indoor+outdoor）提升跨场景泛化。摘要报告同场景训练/测试下较 ACRNet 最高改善 3.42 dB NMSE，跨场景使用时优于单场景训练的基线。

### 3.5 跨流、跨极化与混合编码（2025–2026）

**CASA-TransNet** [14]（Han et al.，IEEE/CIC ICCC 2025）针对双极化 massive MIMO，用两条流分别处理水平/垂直极化分量，通过交换 query embedding 实现跨流 cross-attention，再用 self-attention 建模长程依赖。摘要报告较 SOTA 提升 3.47–5.74 dB；该结果是双极化信道协议，非 COST2100 单极化 32×32。

**TCLNet** [15]（Yang et al.，arXiv 2026）把有损压缩（Transformer-CNN 混合）与无损压缩（语言模型/因子化模型自适应切换 context-aware 与并行编码）统一在一个框架内，并以 LLM 作零样本 CSI 无损压缩器。作者报告真实与仿真数据集上最高 5 dB 的增益，重点是率-失真-复杂度折衷而非单一 NMSE。

**Ansarifard et al.** [16]（arXiv 2025）把空间相关性引导的注意力与量化、熵感知训练结合，学习熵优化的隐表示以降低反馈比特率。摘要称在室内/室外基准上平均超越现有端到端方案 21.5%/18.9%，但未说明该百分比对应的指标口径，本文不将其解读为 dB 增益。

### 3.6 相邻工作与方法来源

**CSI-GPT** [17]（Zeng et al.，arXiv 2024）用 Swin Transformer 构建 SWTCAN，联合设计导频、下行估计与上行反馈；再用 GPT 与 federated-tuning 做在线微调，并以 VAE 生成 CSI 样本缓解数据不足。它属于联合获取而非纯反馈压缩，不能与纯反馈方法同表比较。

**CSformer** [18]（Ye et al.，IEEE Transactions on Image Processing 2023）是图像压缩感知方法：自适应分块采样后，测量值分别进入 CNN stem 与 Transformer stem，并在不同分辨率融合，另用渐进策略与窗口 Transformer 控制复杂度。它不是 CSI 方法，但 STNet 明确把其双 stem 解码器作为设计来源 [9]，因此在 CSI 文献中常被引用为混合架构依据。

## 4. 对比

表 1 汇总设计层面的可核查信息。FLOPs 来自各论文自报，统计工具与是否含编码器/解码器并不统一，只作量级参考。

| 方法（年） | 注意力作用域 | 报告复杂度 | 协议/备注 |
|---|---|---|---|
| TransNet [6]（2022） | 全注意力，token 为角-时延矩阵行向量 | 35.72M FLOPs @1/4；33.75M @1/64 | COST2100，1000 epoch |
| STNet [9]（2022） | 8×8 窗口 LSA + 下采样 KV 的 GSA | 5.22M @1/4 indoor；编码器 2.09M | COST2100，1000 epoch |
| CSIFormer [8]（2022） | 卷积 + 窗口自注意力 | 未核实 | 侧重 CR=1/64 |
| SwinCFNet [10]（2024） | 移位窗口，两阶段层级 | 小配置参数为 TransNet 的 78.48% | COST2100，1000 epoch |
| NEFT-Hybrid [12]（2025） | 编码器无注意力，解码器层级 ViT | 编码端复杂度最多 −64% | 近场 XL-MIMO，非 COST2100 |

表 2 列 COST2100、32×32 的 NMSE。TransNet 与 SwinCFNet 为原文 1000 epoch 协议；前三行取自 SwinCFNet Table I 的汇总，原始训练预算未逐一核实。所有数值均为训练后测试集报告，未说明是否 checkpoint 复评。

| 方法 | Indoor 1/4 | Outdoor 1/4 | Indoor 1/16 | Outdoor 1/16 | 数值来源 |
|---|---|---|---|---|---|
| CsiNet [1] | −17.36 | −8.75 | −8.65 | −4.51 | 二手汇总 |
| Attention-CsiNet [2] | −20.29 | −10.43 | −10.16 | −6.11 | 二手汇总 |
| DS-NLCsiNet [3] | −24.99 | −12.09 | −12.93 | −4.98 | 二手汇总 |
| TransNet [6] | −32.38 | −14.86 | −15.00 | −7.82 | 原文/作者仓库 |
| SwinCFNet [10] | −28.94 | −21.29 | −15.12 | −11.50 | 原文 Table I |

## 5. 小结（文献事实）

1. Indoor 低压缩率（CR=1/4）下全注意力最强：TransNet 报告 −32.38 dB；窗口方法 SwinCFNet 为 −28.94 dB，高 3.44 dB。
2. Outdoor 的排序相反：SwinCFNet 在 CR=1/4 为 −21.29 dB，比 TransNet 的 −14.86 dB 低 6.43 dB；该文同时报告 outdoor 1/4 的 rho 为 0.98，CsiNet 为 0.91。
3. 复杂度差距明显：STNet 自报 5.22M FLOPs，约为 TransNet 35.72M 的 1/7（统计口径不同，仅作量级参考）；原文称 CR=1/16、1/64 时只需 TransNet 的 11.6%、10.8%。
4. 训练预算会改变结论：TransNet 在 400 与 1000 epoch 下 indoor 1/4 相差 3.16 dB，跨论文比较必须先对齐 epoch。
5. 高压缩率下注意力改造的收益结构变化：TransNet+ 相对 TransNet 的 indoor 增益从 1/4 的 0.74 dB 增至 1/64 的 1.91 dB。
6. CR=1/32 时上述模式仍在：TransNet 的 indoor/outdoor 为 −10.49/−4.13 dB，SwinCFNet 为 −9.39/−7.62 dB；indoor 差距缩小到 1.10 dB，outdoor 仍差 3.49 dB。
7. 2024 年后的工作把重点从替换注意力模块转向与量化、熵编码、生成式框架的联合设计（[15][16] 均以量化和熵编码为核心问题），单纯注意力模块的比较已不再是这些论文的主线（本调研判断）。

## 6. 文献明确指出的开放问题

- **量化与熵编码**：TransNet+ 明确指出未考虑码字量化 [7]；Ansarifard et al. 把量化与熵感知训练作为核心 [16]，TCLNet 则把无损熵编码与有损压缩统一设计 [15]。
- **跨场景泛化与数据获取**：AiANet 指出多数模型按单一场景训练、跨场景退化 [13]；TransNet+ 指出实际系统难以采集足够样本、环境变化需要在线训练 [7]。
- **端侧算力**：STNet 指出多头注意力难以部署在功率受限的 UE [9]；NEFT 用无注意力编码器把编码端复杂度降低最多 64% [12]。
- **新阵列形态**：NEFT 指出近场 XL-MIMO 的 CSI 结构与远场 32×32 不同，token 设计需要重做 [12]。
- **复杂度与可复现口径**：STNet 在对比表中以 “-” 标注未公开代码的模型 [9]，各文 FLOPs 统计工具不一致，跨论文复杂度比较缺乏共同基准。

## 7. 参考文献

[1] C.-K. Wen, W.-T. Shih, S. Jin. Deep Learning for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 2018. https://arxiv.org/abs/1712.08919

[2] Q. Cai, C. Dong, K. Niu. Attention Model for Massive MIMO CSI Compression Feedback and Recovery. IEEE WCNC, 2019. https://doi.org/10.1109/WCNC.2019.8885897

[3] X. Yu, X. Li, H. Wu, Y. Bai. DS-NLCsiNet: Exploiting Non-Local Neural Networks for Massive MIMO CSI Feedback. IEEE Communications Letters, 2020. https://doi.org/10.1109/LCOMM.2020.3019653

[4] X. Song, J. Wang, J. Wang, G. Gui, et al. SALDR: Joint Self-Attention Learning and Dense Refine for Massive MIMO CSI Feedback with Multiple Compression Ratio. IEEE Wireless Communications Letters, 2021. https://doi.org/10.1109/LWC.2021.3085317

[5] Y. Xu, M. Yuan, M.-O. Pun. Transformer Empowered CSI Feedback for Massive MIMO Systems. WOCC, 2021. https://doi.org/10.1109/WOCC53213.2021.9602863

[6] Y. Cui, A. Guo, C. Song. TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System. IEEE Wireless Communications Letters, 2022. https://doi.org/10.1109/LWC.2022.3149416

[7] Q. Chen, A. Guo, Y. Cui. Multi-Head Transformer Architecture with Higher Dimensional Feature Representation for Massive MIMO CSI Feedback. Applied Sciences, 2024. https://doi.org/10.3390/app14041356

[8] X. Bi, S. Li, C. Yu, Y. Zhang. A Novel Approach Using Convolutional Transformer for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 2022. https://doi.org/10.1109/LWC.2022.3153085

[9] S. Mourya, S. D. Amuru, K. K. Kuchi. A Spatially Separable Attention Mechanism for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 12(1):40–44, 2023. https://doi.org/10.1109/LWC.2022.3216352

[10] J. Cheng, W. Chen, J. Xu, Y. Guo, L. Li, B. Ai. Swin Transformer-Based CSI Feedback for Massive MIMO. arXiv, 2024. https://arxiv.org/abs/2401.06435

[11] M. Tao, J. Fan, H. Xie, K. Xie. Deep CSI Feedback for FDD Massive MIMO Systems: A Curvelet Learning Approach. arXiv, 2026. https://arxiv.org/abs/2606.17737

[12] T. Mao, H. Li, S. Tan, et al. NEFT: A Unified Transformer Framework for Efficient Near-Field CSI Feedback in XL-MIMO Systems. arXiv, 2025. https://arxiv.org/abs/2509.12748

[13] K. Lou, X. Wu. Attention-Infused Autoencoder for Massive MIMO CSI Compression. IEEE Transactions on Wireless Communications, 2025. https://arxiv.org/abs/2504.12440

[14] Z. Han, H. Zhang, Z. Wan, Z. Gao, Z. Wang. Transformer-Driven CSI Feedback for Dual-Polarized Massive MIMO Systems. IEEE/CIC ICCC, 2025. https://doi.org/10.1109/ICCC65529.2025.11149287

[15] Z. Yang, Q. Yang, S. Tang, T. Yang, Z. Shi. TCLNet: A Hybrid Transformer-CNN Framework Leveraging Language Models as Lossless Compressors for CSI Feedback. arXiv, 2026. https://arxiv.org/abs/2601.06588

[16] M. Ansarifard, M. Rahmani, M. K. Sharma, et al. CSI Compression Beyond Latents: End-to-End Hybrid Attention-CNN Networks with Entropy Regularization. arXiv, 2025. https://arxiv.org/abs/2509.08776

[17] Y. Zeng, L. Qiao, Z. Gao, et al. CSI-GPT: Integrating Generative Pre-Trained Transformer with Federated-Tuning to Acquire Downlink Massive MIMO Channels. arXiv, 2024. https://arxiv.org/abs/2406.03438

[18] D. Ye, Z. Ni, H. Wang, J. Zhang, S. Wang, S. Kwong. CSformer: Bridging Convolution and Transformer for Compressive Sensing. IEEE Transactions on Image Processing, 2023. https://arxiv.org/abs/2112.15299

## 8. 核查记录

**来源等级。** 全文核实：STNet [9]（ar5iv 全文，含复杂度公式、FLOPs 与实验设置）、SwinCFNet [10]（arXiv HTML 全文，含 Table I 的 NMSE/rho）、TransNet+ [7]（MDPI 全文，含增益数值）、TransNet [6]（作者 GitHub 结果表，与论文 Table 1 对应）。摘要核实：[3][5] 经 Semantic Scholar，[8] 经 NASA ADS，[11][12][13][15][16][17][18] 经 arXiv 摘要页，[2] 经 Crossref 元数据，[14] 经作者机构页面。索引片段或转述：[2] 的方法描述与 outdoor 倾向来自 STNet 正文转述，[4] 仅有 Google Scholar 摘要片段，[5] 的性能评价来自 STNet 转述。

**二手数值。** 表 2 前三行（CsiNet、Attention-CsiNet、DS-NLCsiNet）取自 SwinCFNet Table I 的汇总，非本调研复现，原始训练预算与 checkpoint 选择方式未逐一核实；TransNet 与 SwinCFNet 两行为原文/作者仓库数值。未做任何跨协议排名。

**未核实项。** STNet 原文 Table I 的 NMSE 数值（可访问 HTML 未渲染表格，故本文不记录）；TransNet+ Table 1 的具体 NMSE（表格为图片）；CSIFormer 的具体 FLOPs 与训练预算；SwinCFNet 的正式发表 venue 与卷期；NEFT、TCLNet、SwinCANet 的同行评审状态与最终卷期；AiANet 的卷期信息（venue 依 Google Scholar 索引）。

**命名核查。** “DS-Net” 未检索到对应的 CSI 反馈论文（arXiv、Google Scholar，2026-11-03），最接近者为 DS-NLCsiNet [3]；“DS-Net” 另有一篇 2026 年边缘-云协同推理论文，与 CSI 无关。CSFormer 与 CSIFormer 分属 [18] 与 [8]，正文已分别标注。

**检索记录。** 日期 2026-11-03。检索词：CSI feedback transformer；attention CSI feedback；spatially separable attention；DS-Net CSI；CSFormer CSI；Swin Transformer CSI feedback；CSI feedback XL-MIMO transformer；transformer CSI compression 2025/2026。工具：arXiv 搜索接口、Google Scholar、Semantic Scholar Graph API、Crossref API、arXiv 导出 API、IEEE Xplore/ADS/作者主页与 GitHub。
