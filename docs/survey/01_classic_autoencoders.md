# 经典自编码器与卷积架构（2018–2021）

> 本文件只做文献调研，不提出新方法。所有数值均注明数据集、系统参数、压缩比与训练预算；
> 不同论文的数值不构成排名，跨协议比较在文中单独说明。

## 1. 范围与检索方法

本文件覆盖 2018 年 CsiNet 发表至 2021 年前后的经典深度学习 CSI 反馈工作：以自编码器为骨架、
以卷积为主要算子的网络，以及围绕它们出现的时序扩展、量化/多码率扩展、全卷积与率失真建模、
多分辨率与轻量化设计。CS-based DL 在本阶段主要有两种形态：一是在随机压缩测量上只学习恢复器
（CsiNet 论文中的 CS-CsiNet 对照），二是把量化与熵编码纳入端到端训练（DeepCMC）。
模型驱动/展开类工作（ISTA-Net 等）属于另一个主题，留给 `04_model_driven_priors.md`。

纳入标准：FDD 下行 CSI 压缩重建、有可核查的结构与实验、以 2018–2021 年为主；
2022 年后只保留说明范式延续或评测问题的节点工作。来源优先级为 arXiv 全文、DOI/Crossref
元数据、作者代码与发布数据；检索日期 2026-11-03，检索词与结果见文末核查记录。

数值规则：只记录在原文、摘要或作者仓库中读到的数值，综述汇编的数字只作线索；
无法确认训练预算或 checkpoint 选择方式的，表中写“未提及”。

## 2. 时间线与范式演化

2018 年 CsiNet [1] 用“卷积编码器 + 全连接瓶颈 + RefineNet 解码器”完成端到端压缩重建；
2019 年 CsiNet-LSTM [2] 引入帧间相关，DeepCMC [3] 用全卷积结构把量化与熵编码纳入率失真训练，
CsiNet+ 预印本 [4] 处理量化与多码率；2020 年 CRNet [5] 用多分辨率卷积块与训练配方刷新同协议结果，
IEICE 的 CsiNet-Plus [6] 讨论截断与反馈噪声；2021 年 CLNet [7] 转向复数输入与轻量化，
CsiFBnet [8] 改为优化波束成形增益，Distributed DeepCMC [9] 扩展到多用户。2022 年的 STNet [10]
把空间可分离注意力引入该任务（本仓库 baseline 之一，细节归入 `02_attention_transformers.md`）。

从这五年看，范式演化集中在四个方向，而不是单一的性能竞赛：

1. **输入表征基本固定。** 几乎所有工作都先把空间-频率域 CSI 做 2D-DFT，截断到前 32 个时延抽头，
   得到 32×32 的角-时延矩阵，再把实部/虚部当作两个通道。差异主要在编码器如何嵌入
   复数物理意义（CLNet 的 1×1 输入层 [7]）。
2. **瓶颈结构从全连接走向全卷积与多分辨率。** CsiNet 的码字由全连接层产生，参数量几乎都集中
   在这一层；DeepCMC 去掉全连接以适配不同天线数/子载波数 [3]；CRNet 用并行多分辨率路径和
   1×9/9×1 卷积分解，在相近 FLOPs 下扩大感受野 [5]。
3. **训练目标从纯 MSE 扩展。** DeepCMC 最小化“熵率 + λ·MSE”的率失真代价 [3]；CsiNet+ 把
   量化失真和偏置网络纳入训练 [4]；CsiNet-LSTM 在 MSE 上叠加时域重建 [2]；CsiFBnet 直接以
   波束成形性能为目标 [8]。
4. **复杂度意识上升。** CRNet 强调“同等计算量下”比较 [5]；CLNet 明确以 FLOPs 和 UE 侧存储为
   约束，报告相对 CRNet 平均少 24.1% 计算量 [7]。

## 3. 代表性工作

### 3.1 CsiNet：范式起点

CsiNet [1] 的编码器是一层 3×3 卷积加一层全连接，解码器是全连接加两个 RefineNet 单元，
最后用 Sigmoid 把输出压到 [0,1]。训练数据由 COST 2100 模型生成：室内 5.3 GHz、室外 300 MHz，
基站 32 根 ULA 天线、1024 个子载波，保留前 32 行后得到 32×32 矩阵；训练/验证/测试为
100k/30k/20k，训练 1000 epoch、学习率 1e-3、batch 200，并按验证损失选模型。
作者称这是首个把 DL 用于 CSI 压缩与恢复的工作；与 LASSO、TVAL3、BM3D-AMP 相比，
CsiNet 在 1/16 等强压缩条件下仍能工作，且单次前向耗时 0.0035 s，而三种 CS 算法为 0.18–0.57 s。
官方仓库复现的 Table I 给出：室内 CR 1/4 为 −17.36 dB / ρ=0.99，室外为 −8.75 dB / 0.91 [12]。

这篇论文同时埋下了后来评测分歧的种子。正文把压缩比写成 γ=M/N，其中 N=NcNt；
但官方代码里输入维度是 `2×32×32=2048`，CR 1/4 对应 `encoded_dim=512` [12]。
也就是说，“1/4”在正文与实现中的分母并不一致（1024 个复数元素 vs 2048 个实数）。
CsiNet+ 后来明确写成 CR=2NtNc/M [4]，可以看作对这一记法的修正。

### 3.2 时序、量化与多码率：CsiNet-LSTM、CsiNet+

CsiNet-LSTM [2] 把连续 T=10 个 CSI 帧编成一组：第一帧用 1/4 的高码率，其余帧用更低的码率，
码字拼接后送入三层 LSTM 做时域细化。实验用 COST 2100、Nc=256、32×32 截断，室内/室外
训练/验证/测试为 75k/12.5k/12.5k，训练 500–1000 epoch。原文报告室内 CR 1/16 为 −23.06 dB /
ρ=0.99，同表的 CsiNet 为 −10.59 dB / 0.95；室外为 −9.86 dB / 0.95 对 −3.60 dB / 0.75。
需要注意，这里的 CsiNet 是在 Nc=256 的数据上重训的，而 CsiNet 原文是 Nc=1024，
两者不能直接与 CsiNet 论文的 −8.65 dB（室内 1/16）比较。

CsiNet+ [4] 的目标是部署而非单纯精度：编码器改用两个 7×7 卷积，去掉最后一个 RefineNet 之后的
卷积层，并在全连接后加一层 7×7 卷积改善初始估计；量化采用 μ-law 非均匀量化加 offset 网络，
训练分三阶段（无量化预训练、offset 网络、固定编码器微调）。多码率方面，SM-CsiNet+ 串联复用
全连接层，PM-CsiNet+ 从高压缩码字中截取低压缩码字，作者报告 UE 侧参数量分别减少 38.0% 和 46.7%。
主网络训练 1000 epoch、batch 200、初始学习率 1e-3，20 epoch 不降则减半。另有一篇 IEICE 的
“CsiNet-Plus” [6] 讨论截断与反馈噪声，与 CsiNet+ 不是同一工作。

### 3.3 全卷积与率失真：DeepCMC

DeepCMC [3] 去掉了全连接层，编码器是三层卷积（9×9、5×5、5×5）加下采样，解码器对称上采样
并带两个残差块；编码器输出经均匀标量量化后，用 CABAC 做熵编码，训练损失是
“量化后特征的负对数似然（熵率）+ λ·MSE”。因此压缩单位不是 CR，而是
“每信道维度比特数”（bpc）；λ 控制折中点。
实验用 COST 2100 室内、Nc=256、Nt=32，80k 训练 / 20k 测试，作者报告在低于 0.16 bpc 时
NMSE 约 −13 dB、ρ=0.98，并称在相同比特率下比 CsiNet 好 4–6 dB。
无全连接层使同一模型可迁移到不同 Nc/Nt；分布式版本 [9] 利用邻近用户信道的相关性做联合压缩。
其意义是把“反馈多少比特”变成可训练的率失真问题。

### 3.4 多分辨率与训练配方：CRNet

CRNet [5] 的编码器有两条并行路径（一条三级卷积、一条单层 3×3），解码器的 CRBlock 也是
双分辨率并行路径加 1×1 融合与恒等支路；它用 1×9 与 9×1 串联代替大核，在保持感受野的同时
控制 FLOPs。更关键的是训练配方：warmup 30 epoch 后余弦退火，学习率 2e-3 降到 5e-5，
训练 2500 epoch，batch 200。原文在同一张表里给出三组对照：CsiNet（1000 epoch，恒定 1e-3）
室内 CR 1/4 为 −17.36 dB、室外 −8.75 dB；CRNet 在同样 1000 epoch 恒定学习率下为
−21.17 / −10.42 dB；换成余弦配方、2500 epoch 后为 −26.99 / −12.71 dB。
作者还用 100/500/1000/2500/5000 epoch 的消融说明训练长度本身会显著改变结果。

这组对照同时改变了架构与训练预算：同等 1000 epoch 下 CRNet 比 CsiNet 好约 3.8 dB（室内）；
2500 epoch 余弦配方下差距为约 9.6 dB [5]。因此两个数字的差值不能全部归因于架构。
CRNet 也批评部分工作“连基本训练设置都不写”。

### 3.5 轻量化与复数输入：CLNet

CLNet [7] 的出发点是 UE 侧算力与存储受限：它用 1×1 逐点卷积把同一复数的实部/虚部显式绑定，
避免 3×3 卷积把相邻复数的实虚部混在一起；编码器加 SE 通道注意力与 CBAM 空间注意力，
解码器把 CRBlock 的 1×9 卷积缩到 1×3，并用 hard-Sigmoid 代替 Sigmoid。
实验沿用 CsiNet 发布的数据（100k/30k/20k，32×32×2），1000 epoch、batch 200、固定随机种子。
作者报告相对 CRNet 平均精度提升 5.41%、FLOPs 减少 24.1%。

这条结论必须连同脚注读：CLNet 说明它按官方代码复现 CRNet 时只训 1000 epoch，
而 CRNet 论文的更高数字来自 2500 epoch。也就是说，“比 CRNet 好 5.41%”比较的是
同预算复现的 CRNet，而不是 CRNet 原文的最优数字。基线的重训预算会直接影响结论。

### 3.6 任务导向的反馈：CsiFBnet

CsiFBnet [8] 不再以 NMSE 为目标，而是让解码器直接输出波束成形向量，以波束成形性能增益为损失，
并用无监督方式训练。单小区版本是标准自编码器；多小区版本要同时反馈期望信道与干扰信道两类 CSI。
作者报告相对传统 DL 反馈方法有性能提升和复杂度下降。它的意义不在于刷新 NMSE 表，而在于指出
NMSE/ρ 只是波束成形性能的代理指标：当目标函数换成 BF 增益时，网络排序可能改变。
这一点与后来“任务导向/语义 CSI 反馈”的方向一致，但在 2018–2021 年仍属少数。

### 3.7 2022 年之后的走向（仅作定位）

2022 年，Guo 等的综述 [11] 把 DL-based CSI 反馈分为新网络设计、多域相关、比特流生成、
联合设计、实际部署等六类，并把训练数据收集、在线训练、复杂度、泛化、标准化列为实际问题。
2024 年 Liu 等 [13] 处理“每个场景都要一份大数据集、多模型占用存储/更新带宽”的部署障碍，
提出轻量翻译模块、球面归一化和小数据增强。2026 年的 CFM-Bench [14] 指出评测流程在数据、
射频配置、划分、适配、任务定义和指标上各不相同，无法可复现地横向比较；2026 年的 SpikingCSINet [15]
仍以 COST 2100 为实验平台，说明 2018 年建立的数据与基线体系至今仍是公共参照。

## 4. 数据集与评测约定的起源与差异

### 4.1 COST 2100：是信道模型，不是固定数据集

COST 2100 [16] 是几何随机信道模型（GSCM），可复现 MIMO 信道在时间、频率、空间上的统计特性；
它本身不规定子载波数、天线数、采样位置和样本量。CsiNet 给出了一套具体实例化：
室内 5.3 GHz、20 m 见方；室外 300 MHz、400 m 见方；32 根 ULA、1024 个子载波，
截断到 32×32，生成 150k 样本并按 100k/30k/20k 划分 [1]。
所谓“使用 CsiNet 的数据集”，通常指复用作者发布的 MAT 文件；CLNet 明确沿用该数据 [7]，
Guo 的综述也以它为统一比较基础 [11]。

但“COST 2100 + 32×32”并不足以固定协议：CsiNet-LSTM 用 Nc=256、T=10、不同的训练/验证/测试
划分 [2]；DeepCMC 用 Nc=256、80k/20k [3]；CRNet 与 CLNet 用 Nc=1024、100k/30k/20k [5][7]。
同一方法名在不同 Nc 下的 NMSE 可以差约 2 dB（例如 CsiNet 室内 1/16：Nc=1024 时 −8.65 dB [1]，
在 CsiNet-LSTM 的 Nc=256 数据上重训为 −10.59 dB [2]）。因此，“同数据集”只在复用同一批文件、
同一 Nc 与同一划分时才成立。

### 4.2 DeepMIMO：另一类数据来源

DeepMIMO [17] 是基于 Remcom Wireless InSite 射线追踪的通用数据集，面向 mmWave/massive MIMO，
通过场景与参数集定义可复现地生成信道。它与 COST 2100 的生成机制（几何随机模型）、频段与
场景假设都不同。本调研判断：DeepMIMO 上的 CSI 反馈 NMSE 与 COST 2100 上的结果不可直接比较。

### 4.3 NMSE、ρ 与 CR：名字统一、实现不统一

CsiNet 的 NMSE 是 E{‖H−Ĥ‖²/‖H‖²}，ρ 是先对子载波平均、再对样本求期望的归一化内积 [1]。
官方代码先把预测与
真值减 0.5（撤销 [0,1] 归一化），在 32×32 角-时延矩阵上算 NMSE；ρ 则补零做 FFT，在 `HF_all`
的 125 个子载波上算 [12]。同一篇论文的两个指标已不同域，且 NMSE 依赖“减 0.5”，直接在 [0,1]
张量上算会得到不同数值。CRNet、CsiNet-LSTM、CLNet 则把 NMSE 写成角-时延域 `Ha` 与 `Ĥa` 的
归一化误差 [2][5][7]，是否做同样反归一化未统一说明；DeepCMC 用熵编码后的实际比特数除以信道
维度（bpc）[3]。再加上 CR 分母可能是 1024 或 2048，“CR 1/4、NMSE −20 dB”未必指同一件事。

### 4.4 训练预算与 checkpoint 选择

已核查的原文中，CsiNet 按验证损失选模型 [1]；CRNet 训练 2500 epoch 并报告该预算下的结果 [5]；
CLNet 训练 1000 epoch、固定种子，并明确用 1000 epoch 复现的 CRNet 作基线 [7]；
CsiNet-LSTM 的训练预算随收敛情况在 500–1000 epoch 间调整 [2]。
这些论文大多没有说明是否按测试集指标挑选 checkpoint；本仓库 baseline 则用
`select_best: "all"`（test ρ + test NMSE）保存最优模型，跨论文对比时须注明。

### 4.5 数值对照（各自协议内自洽，禁止跨行排名）

| 工作 | 数据与系统 | CR / 码率 | 训练预算 | 原文报告值 | checkpoint 选择 |
|------|------------|-----------|----------|------------|-----------------|
| CsiNet [1][12] | COST 2100 室内/室外，Nt=32，Nc=1024，32×32 | 1/4 | 1000 epoch，batch 200 | 室内 −17.36 dB / ρ 0.99；室外 −8.75 dB / 0.91 | 验证损失最优 |
| CsiNet-LSTM [2] | COST 2100，Nt=32，Nc=256，T=10 | 1/16 | 500–1000 epoch，batch 100 | 室内 −23.06 dB / 0.99；室外 −9.86 dB / 0.95 | 未提及 |
| CRNet [5] | COST 2100，Nt=32，Nc=1024，32×32 | 1/4 | 2500 epoch，warmup+cosine | 室内 −26.99 dB；室外 −12.71 dB | 未提及 |
| DeepCMC [3] | COST 2100 室内，Nt=32，Nc=256 | <0.16 bpc | 率失真训练，正文未见 epoch 数 | 约 −13 dB / ρ 0.98 | 未提及 |
| CLNet [7] | COST 2100，Nt=32，Nc=1024，32×32 | 1/4 | 1000 epoch，batch 200，固定种子 | 摘要相对值：较复现 CRNet +5.41% 精度、−24.1% FLOPs（绝对 NMSE 见原文表 I，本文件未抽取） | 未提及 |

### 4.6 评测差异的来源归纳

| 维度 | 具体差异 | 直接后果 |
|------|----------|----------|
| 数据 | COST 2100 的 Nc、样本量、划分不同；DeepMIMO 属另一类生成机制 | “同数据集”往往只对复用同一批 MAT 文件成立 |
| CR 定义 | 分母 1024 vs 2048；是否把量化比特计入；DeepCMC 用 bpc | 相同 CR 标签对应不同码字长度/比特数 |
| NMSE | 角-时延域 vs 空间-频率域；是否减 0.5 反归一化 | 同一模型可算出不同的 dB 值 |
| ρ | 频域 125 子载波（官方代码）vs 其他域/子载波数 | ρ 与 NMSE 不一定同域，读者容易误读 |
| 训练 | 1000 vs 2500 epoch；恒定 vs 余弦；warmup 有无 | CRNet 的消融显示仅训练配方可带来数 dB 差异 |
| 选择 | 验证损失 vs 固定预算下的报告值 vs 未说明；是否用测试集选模型 | 报告值可能偏向乐观 |

## 5. 小结（只陈述文献事实）

- 2018–2021 的主流范式是“角-时延域 32×32 实数两通道输入 + 端到端自编码器”，
  变化集中在瓶颈与解码器 [1][5][7]。
- CsiNet 的卷积+全连接+RefineNet 骨架、CsiNet+ 的量化/多码率、DeepCMC 的全卷积率失真、
  CRNet 的多分辨率块，是后续反复引用的四个设计参照 [1][3][4][5]。
- 训练配方已成为结果的一部分：同一 CRNet 在 1000 epoch 恒定学习率与 2500 epoch 余弦配方下
  分别为 −21.17 与 −26.99 dB（室内 1/4）[5]。
- 基线常被按新论文自己的预算重训：CLNet 用 1000 epoch 复现的 CRNet 作对照，
  而 CRNet 原文报告 2500 epoch [5][7]。
- COST 2100 是模型而非固定数据集；只有复用同一批文件、同一 Nc 与划分时，
  “同数据集”才成立 [1][2][3][7][16]。
- NMSE/ρ 公式写法一致，但计算域、反归一化与 CR 分母不统一 [1][3][5][12]。
- 2022 年综述与 2026 年 benchmark 都把“数据/划分/指标不一致”列为可复现比较的障碍 [11][14]。

## 6. 文献明确指出的开放问题

1. **训练与报告规范。** CRNet 指出部分工作未交代基本训练设置，并用 epoch 消融说明训练长度的影响 [5]。
2. **数据与场景依赖。** Guo 等把训练数据收集、在线训练、复杂度、泛化与标准化列为实际问题 [11]；
   Liu 等指出每个场景都需规模可观的数据集，多模型还带来存储与更新带宽开销 [13]。
3. **跨流程可比性。** CFM-Bench 指出评测在数据、射频配置、划分、适配流程、任务定义与指标上
   各不相同，无法可复现比较 [14]。
4. **指标与任务目标。** CsiFBnet 以波束成形增益为目标，说明 NMSE/ρ 只是代理指标 [8]。
5. **标准化口径。** Guo 等单列讨论标准化影响 [11]；本文件未核查 3GPP 文档原文，不给具体结论。

## 7. 参考文献

[1] C.-K. Wen, W.-T. Shih, S. Jin. Deep Learning for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 7(5): 748–751, 2018. https://doi.org/10.1109/LWC.2018.2818160 ；arXiv: https://arxiv.org/abs/1712.08919

[2] T. Wang, C.-K. Wen, S. Jin, G. Y. Li. Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels. IEEE Wireless Communications Letters, 8(2): 416–419, 2019. https://doi.org/10.1109/LWC.2018.2874264 ；arXiv: https://arxiv.org/abs/1807.11673

[3] Q. Yang, M. Boloursaz Mashhadi, D. Gündüz. Deep Convolutional Compression for Massive MIMO CSI Feedback. IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2019, pp. 1–6. https://doi.org/10.1109/MLSP.2019.8918798 ；arXiv: https://arxiv.org/abs/1907.02942

[4] J. Guo, C.-K. Wen, S. Jin, G. Y. Li. Convolutional Neural Network-Based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback: Design, Simulation, and Analysis. IEEE Transactions on Wireless Communications, 19(4): 2827–2840, 2020. https://doi.org/10.1109/TWC.2020.2968430 ；arXiv: https://arxiv.org/abs/1906.06007

[5] Z. Lu, J. Wang, J. Song. Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System. IEEE International Conference on Communications (ICC), 2020, pp. 1–6. https://doi.org/10.1109/ICC40277.2020.9149229 ；arXiv: https://arxiv.org/abs/1910.14322

[6] F. Liu, X. He, C. Li, Y. Xu. CsiNet-Plus Model with Truncation and Noise on CSI Feedback. IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, E103.A(1): 376–381, 2020. https://doi.org/10.1587/transfun.2019EAL2123

[7] S. Ji, M. Li. CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback. IEEE Wireless Communications Letters, 10(10): 2318–2322, 2021. https://doi.org/10.1109/LWC.2021.3100493 ；arXiv: https://arxiv.org/abs/2102.07507

[8] J. Guo, C.-K. Wen, S. Jin. Deep Learning-Based CSI Feedback for Beamforming in Single- and Multi-Cell Massive MIMO Systems. IEEE Journal on Selected Areas in Communications, 39(7): 1872–1884, 2021. https://doi.org/10.1109/JSAC.2020.3041397 ；arXiv: https://arxiv.org/abs/2011.06099

[9] M. Boloursaz Mashhadi, Q. Yang, D. Gündüz. Distributed Deep Convolutional Compression for Massive MIMO CSI Feedback. IEEE Transactions on Wireless Communications, 20(4): 2621–2633, 2021. https://doi.org/10.1109/TWC.2020.3043502 ；arXiv: https://arxiv.org/abs/2003.04684

[10] S. Mourya, S. Amuru, K. K. Kuchi. A Spatially Separable Attention Mechanism for Massive MIMO CSI Feedback. arXiv:2208.03369, 2022. https://arxiv.org/abs/2208.03369

[11] J. Guo, C.-K. Wen, S. Jin, G. Y. Li. Overview of Deep Learning-Based CSI Feedback in Massive MIMO Systems. IEEE Transactions on Communications, 70(12): 8017–8045, 2022. https://doi.org/10.1109/TCOMM.2022.3217777 ；arXiv: https://arxiv.org/abs/2206.14383

[12] W.-T. Shih, C.-K. Wen. Python code for “Deep Learning for Massive MIMO CSI Feedback”. GitHub repository, 2018. https://github.com/sydney222/Python_CsiNet

[13] Z. Liu, J. Wang, W. Xu, Z. Ding. Deep Learning for Efficient CSI Feedback in Massive MIMO: Adapting to New Environments and Small Datasets. IEEE Transactions on Wireless Communications, 23(9): 12297–12312, 2024. https://doi.org/10.1109/TWC.2024.3390583 ；arXiv: https://arxiv.org/abs/2211.14785

[14] Y. Gao, W. Yu, J. Jiang, Y. Li, X. Guo, S. Xu. CFM-Bench: A Unified Multi-Domain, Multi-Task Benchmark for Channel Foundation Models. arXiv:2607.14975, 2026. https://arxiv.org/abs/2607.14975

[15] Y. Liu, et al. Massive MIMO CSI Feedback with Spiking Neural Networks. arXiv:2605.12434, 2026. https://arxiv.org/abs/2605.12434

[16] L. Liu, C. Oestges, J. Poutanen, K. Haneda, P. Vainikainen, F. Quitin, F. Tufvesson, P. De Doncker. The COST 2100 MIMO Channel Model. IEEE Wireless Communications, 19(6): 92–99, 2012. https://doi.org/10.1109/MWC.2012.6393523

[17] A. Alkhateeb. DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications. Information Theory and Applications Workshop (ITA), 2019. https://arxiv.org/abs/1902.06435

## 8. 核查记录

**来自原文全文（ar5iv/HTML 或 arXiv 正文）**

- CsiNet [1]：系统模型、COST 2100 参数（5.3 GHz/300 MHz、Nt=32、Nc=1024）、100k/30k/20k、
  1000 epoch、验证损失选模型、NMSE/ρ 定义与运行时间。
- CRNet [5]：结构、150k 划分、2500 epoch、warmup/余弦、同表三组数值、epoch 消融与对报告不完整的批评。
- CsiNet+ [4]：结构修改、μ-law 量化与 offset 网络、三段训练、38.0%/46.7%、CR=2NtNc/M。
- DeepCMC [3]：全卷积、CABAC、率失真损失、Nc=256、80k/20k、−13 dB/0.98、bpc 口径。
- CsiNet-LSTM [2]：T=10、Nc=256、75k/12.5k/12.5k、500–1000 epoch、表 I 数值。
- CLNet [7]：复数输入、SE/CBAM、1000 epoch/固定种子、5.41%/24.1%、CRNet 2500 epoch 脚注。
- Guo 综述 [11]：摘要与第 I–IV-A1 节（六类划分、实际问题清单、表 I 的汇编性质）。

**来自摘要（未读全文）**

- Distributed DeepCMC [9]、CsiFBnet [8]、DeepMIMO [17]、Liu 等 2024 [13]、CFM-Bench [14]、
  SpikingCSINet [15]、CsiNet-Plus [6]、COST 2100 [16]；STNet [10] 仅核对题录。其中 [6] 只读摘要页。

**来自作者仓库（一手代码/README，不是论文 PDF）**

- CsiNet README 的 Table I 复现值；`CsiNet_train.py` 中 `encoded_dim=512` 对应 CR 1/4、
  `img_total=2*32*32`、评测前减 0.5、`HF_all` 按 32×125 使用；
  本仓库 `data/*.mat` 的形状为 `HT` (N,2048)、`HF_all` (N,4000)。

**二手来源**

- Guo 综述 [11] 的表 I 汇编了多篇论文的 NMSE；本文件未引用该表数字作为原文结果，
  正文数值均回到原文或作者仓库核对。未使用博客、百科或二手解读中的数值。

**检索词与日期**

- 检索日期：2026-11-03。
- 英文检索词：massive MIMO CSI feedback；CsiNet；CsiNet+ multiple-rate；CRNet multi-resolution；
  DeepCMC compressive CSI feedback；STNet；CSI feedback benchmark/reproducibility；CSI feedback survey。
- arXiv API：`massive MIMO CSI feedback`（2025-01-01 至 2026-11-03）、`CSI feedback benchmark evaluation`，
  以及 CsiNet-LSTM、DeepMIMO、STNet、Guo 综述的标题检索。
- 元数据核对：Crossref `query.bibliographic` 核对 DOI、卷期、页码；dblp 因反爬未使用。

**已知不确定性**

- CsiNet 论文 Table I 在 ar5iv 中是图片，本文件的 −17.36/−8.75 等数值来自作者仓库 README；
  两者应一致，但未逐格比对 PDF 原表。
- CRNet、CLNet、CsiNet-LSTM 的 NMSE 是否做了与官方 CsiNet 相同的“减 0.5”反归一化，
  原文未明确说明；本文件只陈述“在角-时延域计算”，不推断数值等价性。
- CFM-Bench 与 SpikingCSINet 只核对了 arXiv 摘要页，引用范围限于摘要陈述。
- COST 2100 的“室内 20 m / 室外 400 m、Nc=1024”是 CsiNet 的实例化参数，不代表模型本身的规定。
