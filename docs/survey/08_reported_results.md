# 跨论文已报告结果与可复现性

本文汇总 COST2100 indoor / outdoor 场景下、压缩比 1/4–1/64 的已报告 NMSE(dB)、ρ 与复杂度数值，
并整理文献中**明确写出**的评测不一致与复现困难。目的不是给出跨论文排名，而是说明：在现有报告习惯下，
哪些数值可以直接对照，哪些不能。

## 一、范围与检索方法

**范围**：以自编码器类 CSI 反馈为主线，包含 2018 年 CsiNet 之后的卷积/注意力/Transformer 方法、
比特级与量化方法、以及 2025–2026 年的能效与基准类工作。只纳入报告 COST2100 数值、或直接讨论评测协议的工作；
3GPP 码本、AI/ML 空口标准化只在涉及"如何选 baseline、如何报复杂度"时作为相邻证据引用。

**检索方式**：arXiv API 与 arXiv HTML 全文、Crossref 元数据（核对 DOI、卷期、页码）、论文作者官方仓库、
出版社 DOI 页。检索日期 2026-11-03。主要检索词：`CSI feedback`、`massive MIMO CSI feedback`、
`multi-resolution CSI feedback`、`spatially separable attention CSI feedback`、`COST2100`、
`CSI feedback benchmark`、`reproducible CSI feedback`、`CSI feedback survey`。

**记录规则**（与本文所有表格一致）：

1. 数值只取自原文正文/表格，或作者官方仓库中明确写为"论文报告值"的表格；
2. 原文表格为图片、无法机读时，标注"未核实"，不用二手数字补齐；
3. 凡属转引（后续论文对照表中列出的他人结果），单独标注"转引"，不与原文结果混排；
4. 不做跨协议排名；同一张表内只做同协议内的纵向对照。

## 二、一个 COST2100 结果要写清哪些字段

阅读原文可以发现，同一句"indoor CR=1/4 达到 −29 dB"背后至少有七个可变字段，缺一项即不可直接比较。

- **数据版本与生成参数**：COST2100 indoor（5.3 GHz 皮蜂窝）与 outdoor（300 MHz 乡村）由信道模型生成，
  并非固定数据集文件。CsiNet 给出 32 天线、1024 子载波、保留前 32 行，共 150,000 样本按
  100,000 / 30,000 / 20,000 划分 [1]；后续工作多数沿用这一划分 [3][4][5][7]，但生成脚本、
  随机种子与是否重新生成很少说明。个别工作直接引用第三方镜像数据集（如 Kaggle 上的 COST2100）[11]。
- **预处理与归一化**：角度-时延域 2D-DFT、截断 32 行是共同起点；但 CsiNet 将输入归一化到 [0,1]
  并在输出端用 sigmoid [1]，而 2026 年的 SpikingCSINet 把输入缩放到 [−25,25] 以驱动脉冲活动 [10]。
  归一化区间不同，NMSE 的数值尺度就不同。
- **压缩比定义**：码字维度 M 与实数化维度 `2N_cN_t` 之比是压缩率，其倒数是压缩倍数，二者都常被写作"CR" [3][7]；
  CsiNet 原文用 `γ=M/N`（N=N_cN_t）[1]，与实数化维度差一个 2；本文按仓库习惯用 CR=1/4 表示 M=512。
- **指标聚合方式**：通行定义为 `NMSE = E{||H−Ĥ||²/||H||²}` 再取 dB [1][4]；但也有工作报告
  "线性域 NMSE 降低 17.72%"这类百分比 [11]，或逐样本 dB 后平均。三者在低 NMSE 区间可差 1 dB 以上。
- **ρ 的计算范围**：CsiNet 原文将 ρ 定义为对全部 N_c 个子载波求平均 [1]，但其官方代码说明：
  受计算资源限制，实际只比较了前 125 个子载波，`DATA_HtestFin_all.mat` 的 4000 维即 125×32 [2]。
  这直接决定了 ρ 的可比性（本仓库使用同名文件）。
- **训练预算**：CsiNet 1000 epoch 恒定学习率 [1]；CRNet 主张 2500 epoch + warmup-cosine [4]；
  CLNet、STNet、STQENet、SpikingCSINet 回到 1000 epoch [5][7][10][11]。
- **checkpoint 选择**：CsiNet 明确写"选验证集损失最小的参数组" [1]；其余多数论文正文未说明
  报的是最后一个 epoch、验证最优还是测试最优。这一点列入第七节。

另有两个常被忽略的字段：**是否量化**（全精度浮点码字 vs 2–8 bit 码字）与**复杂度口径**
（只算 encoder、算全网、是否计入 BN）。二者直接决定"轻量"结论是否成立。

## 三、时间线与代表性工作

### 3.1 2018–2020：自编码器与训练策略

**CsiNet**（Wen, Shih, Jin；IEEE WCL 2018）[1] 是这一任务的起点：encoder 为单层 3×3 卷积加全连接，
decoder 为全连接加两个 RefineNet，端到端 MSE 训练，1000 epoch、batch 200、学习率 0.001。它在
CR=1/4 的 indoor 上把 NMSE 从 CS 方法的 −14.87 dB（TVAL3）降到 −17.36 dB，并同时报告 ρ；
这是本文核查范围内**唯一**同时给出 NMSE 与 ρ 全表的论文。

**CsiNet-LSTM**（Wang, Wen, Jin, Li；IEEE WCL 2019）[9] 在 CsiNet 上加入 LSTM 利用相邻时隙相关性，
摘要称其在压缩比-复杂度权衡上更优；它把评测从"单帧重建"扩展到"时变信道"，
也引入时隙间隔、速度等新的协议维度。

**CsiNet+**（Guo, Wen, Jin, Li；IEEE TWC 2020）[3] 用 7×7 大核、去掉 RefineNet 之后的卷积层、
并加入 μ-law 非均匀量化与 offset 网络。其 encoder 在 CR=1/4 时为 1,049,500 参数，其中全连接层
1,048,576，占 99.9%；这解释了为什么"参数量"在这类网络中几乎由码字维度决定，而非网络结构。

**CRNet**（Lu, Wang, Song；IEEE ICC 2020）[4] 提出多分辨率 CRBlock 与卷积分解，并第一次把
训练方案本身当作贡献：warmup + cosine 退火、2500 epoch。它的 Table II 是本领域少见的
"epoch 预算消融"（见第五节），也是后文多起不可比问题的源头。

### 3.2 2021–2023：复数输入、注意力与轻量化

**CLNet**（Ji, Li；IEEE WCL 2021）[5] 用 1×1 卷积把实部/虚部成对嵌入（"forged complex-valued input"），
配合 SE 与 CBAM 注意力、hard-sigmoid 激活；官方仓库给出逐 CR 的 NMSE、参数量与 FLOPs [6]。

**TransNet**（Cui, Guo, Song；IEEE WCL 2022）[8] 采用两层 full attention，是 2022 年前后
Transformer 路线的性能参照；其代价是复杂度，STNet 以"约 11%–15% 的 FLOPs 达到其 97%–128% 的相对性能"
来定位自己 [7]。

**STNet**（Mourya, Amuru, Kuchi；IEEE WCL 2023）[7] 用局部窗口注意力（LSA）加全局子采样注意力（GSA）
构造空间可分离注意力，decoder 采用 CNN/Transformer 双 stem。协议为：100,000/30,000/20,000 划分、
batch 200、1000 epoch、学习率 0.001、Adam、MSE；但逐 CR 的 NMSE 表在 HTML 版本中以图片呈现，
本调研无法机读（见核查记录）。

### 3.3 2024–2026：比特级、能效与基准

**STQENet**（Ansarifard et al.；arXiv 2025）[11] 把量化与 Huffman 熵编码纳入端到端训练，
评测改为 NMSE–BPP 曲线而非固定 CR；为"公平比较"，作者给两个基线也加装了熵瓶颈。

**SpikingCSINet**（Liu, Li；arXiv 2026）[10] 用脉冲神经网络加渐进残差，报告 NMSE 与能耗（μJ），
并在一张表内同时给出 CsiNet、CLNet、CRNet、CsiNet+、TransNet 的全精度结果与 2–6 bit 量化结果；
脚注说明原论文未报量化性能，故用官方代码重训并以 STE 微调。这张表是本文所见**协议最统一**的横向对照，
但其每列究竟是原文值还是重训值，脚注并未逐列交代。

**CFM-Bench**（Gao et al.；arXiv 2026）[12] 是跨 6 个信道域、6 类任务（含 CSI feedback）的基准，
其动机陈述可直接引用："模型各自使用不同的数据、无线配置、划分、适配流程、任务定义与指标，
使可复现比较无法进行"。它并不针对 COST2100，但把"评测管线本身是变量"这一点写成了显式问题。

**AI/ML for Mobile Networks: Release 19**（Wu et al.；arXiv 2026）[13] 从标准化视角列出三个落地障碍：
3GPP 规范下的数据集构造、跨场景/跨仿真器泛化、以及**联合性能-成本 KPI 下的 baseline 选择**；
其案例使用 DeepMIMO 与 QuaDRiGa 而非 COST2100，属于相邻证据。

## 四、已报告数值

四张表分别对应四类可核查来源：原文（表 1、表 2）、作者官方仓库（表 3）、协议字段（表 4）。
除特别注明外，训练/验证/测试划分为 100,000 / 30,000 / 20,000。

**表 1｜CsiNet 原文 Table I（经官方仓库复现表核对）** [1][2]

| CR | indoor NMSE (dB) | indoor ρ | outdoor NMSE (dB) | outdoor ρ |
|----|------------------|----------|-------------------|-----------|
| 1/4  | −17.36 | 0.99 | −8.75 | 0.91 |
| 1/16 | −8.65  | 0.93 | −4.51 | 0.79 |
| 1/32 | −6.24  | 0.89 | −2.81 | 0.67 |
| 1/64 | −5.84  | 0.87 | −1.93 | 0.59 |

注：原文表不含 CR=1/8。CsiNet 报告的是 1000 epoch 恒定学习率、按验证损失选模型的结果 [1]。

**表 2｜CRNet 原文 Table I（NMSE 与 FLOPs）** [4]

| CR | CsiNet indoor | CRNet-const indoor | CRNet-cosine indoor | CsiNet outdoor | CRNet-cosine outdoor | CRNet FLOPs |
|----|---------------|--------------------|---------------------|----------------|----------------------|-------------|
| 1/4  | −17.36 | −21.17 | −26.99 | −8.75 | −12.71 | 5.12 M |
| 1/8  | −12.70 | −13.79 | −16.01 | −7.61 | −8.04  | 4.07 M |
| 1/16 | −8.65  | −10.29 | −11.35 | −4.51 | −5.44  | 3.55 M |
| 1/32 | −6.24  | −8.58  | −8.93  | −2.81 | −3.51  | 3.28 M |
| 1/64 | −5.84  | −6.14  | −6.49  | −1.93 | −2.22  | 3.16 M |

注：`const` 为 1000 epoch 恒定学习率 0.001，`cosine` 为 2500 epoch、30 epoch warmup、2e-3→5e-5。
CsiNet 列与原文表一致（CR=1/8 一行除外，见 5.6），CRNet 列为作者自己训练。

**表 3｜CLNet 官方仓库给出的论文报告值** [6]

| CR | indoor NMSE (dB) | outdoor NMSE (dB) | 参数量 | FLOPs |
|----|------------------|-------------------|--------|-------|
| 1/4  | −29.16 | −12.88 | 2102 K  | 4.42 M |
| 1/8  | −15.60 | −8.29  | 1053 K  | 3.37 M |
| 1/16 | −11.15 | −5.56  | 528.7 K | 2.85 M |
| 1/32 | −8.95  | −3.49  | 266.5 K | 2.58 M |
| 1/64 | −6.34  | −2.19  | 135.4 K | 2.45 M |

注：仓库注明论文 FLOPs 由 fvcore 计算且不含 BN 层，仓库自身打印值由 thop 计算、含 BN，故二者不同 [6]；
参数量为全网（encoder+decoder）。训练为 1000 epoch，并固定随机种子 [5]。

**表 4｜协议字段对照（本文核查范围内可确证者）**

| 论文 | 训练预算 | 模型选择 | 报告指标 | 量化 | 复杂度口径 |
|------|----------|----------|----------|------|------------|
| CsiNet [1] | 1000 ep, lr 1e-3 恒定 | 验证损失最小 | NMSE + ρ | 全精度 | 推理时间；FLOPs 由他人补算 |
| CsiNet+ [3] | 1000 ep（比特级微调 200 ep），平台减半 | 正文未说明 | NMSE | μ-law + offset | encoder 参数量 1,049,500 |
| CRNet [4] | 2500 ep, warmup-cosine | 正文未说明 | NMSE | 全精度 | FLOPs |
| CLNet [5][6] | 1000 ep, 固定种子 | 正文未说明 | NMSE | 全精度 | FLOPs（fvcore，不含 BN）+ 参数量 |
| STNet [7] | 1000 ep, lr 1e-3 恒定 | 正文未说明 | NMSE + FLOPs + 运行时 | 全精度 | FLOPs（encoder 占 40.03%） |
| STQENet [11] | 1000 ep | 正文未说明 | NMSE vs BPP | 2–8 bit + Huffman | BPP |
| SpikingCSINet [10] | 1000 ep, lr 2e-3 cosine | 正文未说明 | NMSE + 能耗 | 全精度 / 2–6 bit | 能耗（μJ, 45 nm CMOS） |

**未能核实的数值**：CsiNet+ 的逐 CR NMSE 以折线图（Fig. 6）给出，非表格 [3]；TransNet 原文的 NMSE 表
本调研未能取得可机读版本；STNet 的逐 CR NMSE 表为图片 [7]。SpikingCSINet 的对照表给出 TransNet 在
indoor 上 −32.38 / −22.91 / −15.00 / −10.49 / −6.08 dB（CR 1/4→1/64），但该列是原文值还是重训值，
脚注未逐列说明，本文只将其视为量级参考，不作为 TransNet 的原文结果 [10]。

## 五、文献明确指出的评测不一致

### 5.1 训练预算不同，结论方向可能反转

CRNet 在正文中直接批评同行："有些工作甚至省略了基本训练方案的描述，这严重损害了其报告性能的可靠性。"
其 Table II 给出同一网络、同一 CR、同一数据下的 epoch 消融（indoor，CR=1/4）：恒定学习率方案在
100 / 500 / 1000 / 2500 / 5000 epoch 上分别为 −18.94 / −17.55 / −21.17 / −22.94 / −21.89 dB，
warmup-cosine 方案为 −21.33 / −23.76 / −25.17 / −26.99 / −27.31 dB [4]。1000 与 2500 epoch
之间相差 1.82 dB，且恒定方案在 5000 epoch 反而退化。CLNet 则给出了
另一侧的证词：它复现 CRNet 官方代码时发现"论文中更高的性能来自 2500 epoch 训练"，而 CLNet 自己在
1000 epoch 下比较 [5]。即两篇论文表格数值来自不同 epoch 预算，直接对照会误判。

### 5.2 指标实现与 ρ 的覆盖范围

CsiNet 官方仓库的自述是本文见到的最具体的实现偏差：ρ 本应对全部子载波平均，实际只用了前 125 个 [2]。
ρ 是波束赋形质量的代理指标，子载波子集不同会改变其绝对值，高 CR 下更明显。
与此同时，ρ 的报告在 2019 年后明显减少：CRNet、CLNet、STNet、STQENet、SpikingCSINet 的表
均只报 NMSE（及 FLOPs/能耗/BPP）[4][5][7][10][11]。对本仓库而言，这意味着"NMSE + ρ"的双指标
在文献中缺乏可对照的靶值。

### 5.3 复杂度口径不可互换

三处证据指向同一问题。其一，CLNet 论文脚注称"CRNet 论文报告的 FLOPs 已被 [CV-3DCNN] 更正" [5]，
即第三方认为原报告的 FLOPs 有误。其二，CLNet 仓库说明论文 FLOPs 用 fvcore、不含 BN，而仓库打印值
用 thop、含 BN [6]——同一模型两套数字。其三，参数量存在"只算 encoder"与"算全网"两种口径：
CsiNet+ 报 encoder 1,049,500 [3]，CLNet 报全网 2102 K（CR=1/4）[6]，而 SpikingCSINet 的参数量表
在 CR=8 下给 CsiNet/CLNet/CRNet 均为 1.05 M 左右 [10]，恰是"全网、且由全连接层主导"的结果。
把这三列并排比较会得到完全错误的"轻量"结论。

### 5.4 数据与预处理的隐性差异

划分比例在 2018 年后高度一致（100k/30k/20k），但生成参数、随机种子、归一化区间并未统一。
SpikingCSINet 明确写出额外的输入处理：输入缩放至 [−25,25]，并对 indoor 施加随机相位旋转
数据增强（K=16），理由是脉冲模型在 indoor 易过拟合 [10]。STQENet 则引用 Kaggle 镜像数据集，
并把子载波数写作 `N_c=32`（与"保留 32 个时延抽头"混淆）[11]。

### 5.5 全精度与比特级结果混排

CsiNet+ 的量化框架需要三阶段训练（预训练、offset 网络、联合微调），比特级模型只微调 200 epoch [3]；
STQENet 直接在 2–8 bit 下评测并画 NMSE–BPP [11]；SpikingCSINet 把"全精度（∞）"与 T/b=2,4,6 并列 [10]。
同一张表里同时出现浮点码字与 4 bit 码字时，数值差异主要来自比特预算而非网络结构。
此外，STQENet 为对齐协议给两个基线加装了熵瓶颈，被改造后的基线已不等于其原文模型 [11]。

### 5.6 "未报告"与版本漂移

STNet 的表格用两种符号区分缺失：`/` 表示原论文未报告，`-` 表示代码未公开、无法自行生成 [7]。
这提示一个常被忽略的事实：对照表中的空白并不都意味着"方法弱"，很多只是"没人测过"。
版本漂移也有实例：同一 arXiv 条目 v1 与 v2 对 CR=1/16 的相对性能表述分别为 101.86% 与 102.86%，
v2 还删去了 v1 中 CR=1/4 的那句比较，并新增了与 CSITransformer 的运行时对比（同时声明后者是在
不同数据集上训练的）[7]。另有转引来源不明的情况：CRNet 表中列出 CsiNet 在 CR=1/8 的
−12.70 / −7.61 dB，而 CsiNet 原文表只有 1/4、1/16、1/32、1/64 四档，该行来源未作说明 [1][4]。

### 5.7 复现困难的其他证词

CFM-Bench 把问题上升为基准设计问题：各模型使用不同的数据、无线配置、划分、适配流程、任务定义与指标，
"使可复现比较无法进行" [12]。Release 19 的综述则把"联合性能-成本 KPI 下的 baseline 选择"列为
三个落地障碍之一，并报告跨仿真器零样本迁移会出现显著退化 [13]。这两条都不是 COST2100 专属，
但说明更换评测管线即可改变结论。

### 5.8 内部记录：本仓库的复现对照（非文献）

作为同一现象的本地样本：本仓库以 1000 epoch、batch 200、`select_best: "all"` 跑完的 baseline 中，
CsiNet indoor CR=1/4 为 −14.45 dB（文献 −17.36 dB），CLNet indoor CR=1/4 为 −27.87 dB（文献 −29.16 dB），
CLNet outdoor CR=1/4 为 −5.90 dB（文献 −12.88 dB，且低于其自身 CR=1/8 的 −8.00 dB，疑为训练稳定性问题）。
选择策略、随机种子与评测时机均与原文不同，因此这些差距不能用来判断谁对谁错，只能说明协议敏感度之高。
详见 `docs/baselines.md`。

## 六、小结

1. 2018–2026 年间，COST2100 的**数据划分**趋于统一（100k/30k/20k），但**训练预算**在 1000 与 2500 epoch
   之间分裂，且多数论文未说明 checkpoint 选择策略。
2. 在数据划分与指标定义一致的前提下，indoor CR=1/4 的 NMSE 从 CsiNet 的 −17.36 dB 降至
   CLNet 的 −29.16 dB，outdoor 从 −8.75 dB 降至 −12.88 dB；但训练预算与选择策略不同 [1][4][6]。
3. 核查范围内，ρ 只在 CsiNet 原文中系统报告（且其官方实现只用了 125 个子载波）；此后多数工作只报 NMSE，
   ρ 逐渐退出对照表 [1][2][4][5][7]。
4. 复杂度数值至少存在三种口径（encoder-only / 全网、fvcore / thop、FLOPs / 能耗 / BPP），
   跨论文并排比较会失真 [3][6][10][11]。
5. 2025–2026 年的新工作倾向于把量化、熵编码、能耗纳入评测，并主动重训基线；
   这提高了单篇论文内部的一致性，但"重训值"与"原文值"的混用又带来新的溯源问题 [10][11]。

## 七、文献明确指出的开放问题

- **训练方案与报告规范**：CRNet 指出省略训练方案描述会损害结果可靠性，并给出 epoch 消融 [4]；
  但在本文核查范围内，未见统一的"最少报告字段"约定。
- **模型选择与测试集使用**：除 CsiNet 明确写按验证损失选模型外 [1]，本文核查的其余论文正文未说明
  checkpoint 是按验证集还是测试集选择；若为后者，报告值会偏乐观。
- **报告值的来源**：上述论文均未说明数值取自训练过程中的最优记录，还是重新加载 checkpoint 后的复评；
  本仓库的记录属前者（`docs/baselines.md` 明示为训练中观察到的最优值，非复评值）。
- **ρ 的实现一致性**：CsiNet 官方实现的 125 子载波限制 [2] 未见后续工作澄清。
- **复杂度口径**：FLOPs 计算工具（fvcore / thop / 是否含 BN）与统计范围（encoder / 全网）未统一，
  第三方更正前人 FLOPs 的情况已经发生 [5][6]。
- **基准与跨域可比性**：CFM-Bench 主张用统一划分、任务定义与指标来消除管线差异 [12]；
  Release 19 综述则强调跨仿真器泛化与联合 KPI 下的 baseline 选择 [13]。二者都指向"单一数据集内的
  排名不足以支撑部署结论"。
- **比特级评测**：全精度、固定比特、NMSE–BPP 三种报告方式并存，尚无一者成为默认 [3][10][11]。

## 八、参考文献

[1] C.-K. Wen, W.-T. Shih, S. Jin. Deep Learning for Massive MIMO CSI Feedback. IEEE Wireless
Communications Letters, 7(5): 748–751, 2018. https://doi.org/10.1109/LWC.2018.2818160

[2] W.-T. Shih, C.-K. Wen. Python_CsiNet（CsiNet 官方实现与 Table I 复现表）. GitHub.
https://github.com/sydney222/Python_CsiNet

[3] J. Guo, C.-K. Wen, S. Jin, G. Y. Li. Convolutional Neural Network-Based Multiple-Rate Compressive
Sensing for Massive MIMO CSI Feedback: Design, Simulation, and Analysis. IEEE Transactions on Wireless
Communications, 19(4): 2827–2840, 2020. https://arxiv.org/abs/1906.06007

[4] Z. Lu, J. Wang, J. Song. Multi-resolution CSI Feedback with Deep Learning in Massive MIMO System.
IEEE ICC 2020, pp. 1–6. https://arxiv.org/abs/1910.14322

[5] S. Ji, M. Li. CLNet: Complex Input Lightweight Neural Network Designed for Massive MIMO CSI Feedback.
IEEE Wireless Communications Letters, 10(10): 2318–2322, 2021. https://arxiv.org/abs/2102.07507

[6] S. Ji. CLNet 官方仓库（含论文报告 NMSE、参数量与 FLOPs 表）. GitHub.
https://github.com/SIJIEJI/CLNet

[7] S. Mourya, S. Amuru, K. K. Kuchi. A Spatially Separable Attention Mechanism for Massive MIMO CSI
Feedback. IEEE Wireless Communications Letters, 12(1): 40–44, 2023. https://arxiv.org/abs/2208.03369
（v1: https://arxiv.org/abs/2208.03369v1）

[8] Y. Cui, A. Guo, C. Song. TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO
System. IEEE Wireless Communications Letters, 11(5): 903–907, 2022.
https://doi.org/10.1109/LWC.2022.3149416

[9] T. Wang, C.-K. Wen, S. Jin, G. Y. Li. Deep Learning-Based CSI Feedback Approach for Time-Varying
Massive MIMO Channels. IEEE Wireless Communications Letters, 8(2): 416–419, 2019.
https://doi.org/10.1109/LWC.2018.2874264

[10] Y. Liu, G. Y. Li. Massive MIMO CSI Feedback with Spiking Neural Networks. arXiv:2605.12434, 2026.
https://arxiv.org/abs/2605.12434

[11] M. Ansarifard, M. Rahmani, M. K. Sharma, K. C. Joshi, G. Exarchakos, A. Burr. CSI Compression Beyond
Latents: End-to-End Hybrid Attention-CNN Networks with Entropy Regularization. arXiv:2509.08776, 2025.
https://arxiv.org/abs/2509.08776

[12] Y. Gao, W. Yu, J. Jiang, Y. Li, X. Guo, S. Xu. CFM-Bench: A Unified Multi-Domain, Multi-Task Benchmark
for Channel Foundation Models. arXiv:2607.14975, 2026. https://arxiv.org/abs/2607.14975

[13] X. Wu, J. Jiang, B. Hu, J. Du, Q. Ye, S. Zhang, F. R. Yu, S. Xu. AI/ML for Mobile Networks:
Release 19 Status and Future Challenges. arXiv:2603.14317, 2026. https://arxiv.org/abs/2603.14317

## 九、核查记录

**来自原文全文（arXiv HTML / ar5iv，逐段读取）**：CsiNet（arXiv:1712.08919v4）的训练设置、
100k/30k/20k 划分、NMSE/ρ 定义与"按验证损失选模型"表述；CsiNet+（arXiv:1906.06007v1）的训练设置、
encoder 参数量 1,049,500 与"全连接层占 99.9%"；CRNet（arXiv:1910.14322v2）的 Table I、Table II、
2500 epoch 与对同行"省略训练方案"的批评；CLNet（arXiv:2102.07507v3）的数据划分、1000 epoch、
固定种子、对 CRNet 使用 2500 epoch 的脚注、以及"CRNet FLOPs 已被更正"的脚注；STNet
（arXiv:2208.03369 v1 与 v2）的协议、表格符号约定与 v1/v2 表述差异；SpikingCSINet（arXiv:2605.12434v1）
的 Table I、Table II、脚注与输入缩放/数据增强说明；STQENet（arXiv:2509.08776v1）的 NMSE–BPP 协议、
"给基线加装熵瓶颈"与 Kaggle 数据集引用。

**来自作者官方仓库**：CsiNet 的 Table I 复现表与"ρ 只取前 125 个子载波"的 Remarks；
CLNet 的 NMSE、参数量、FLOPs 表与 fvcore/thop 说明。

**仅来自摘要**：CFM-Bench [12]、AI/ML Release 19 [13]；二者的具体实验数字未在本文引用。

**转引（非原文）**：TransNet 的逐 CR 数值取自 SpikingCSINet 的对照表 [10]，未取得 TransNet 原文表。

**未能核实**：CsiNet+ 逐 CR NMSE（原文为折线图）；STNet 逐 CR NMSE（表格为图片，HTML 版本不可机读）；
TransNet 原文 NMSE 与 FLOPs；CsiNet 在 CR=1/8 的原始报告值（原文表无此档，CRNet 表中的该行来源不明）。

**本调研判断（非原文陈述）**：SpikingCSINet 正文写"每个场景 10,000 训练样本、30,000 验证、20,000 测试"，
训练集小于验证集，与其引用的 100k/30k/20k 惯例不符，疑为笔误；STQENet 将子载波数写作 `N_c=32`，
与"保留 32 个时延抽头"混淆；5.8 节的"疑为训练稳定性问题"为对 `docs/baselines.md` 已知现象的转述。

**检索词与日期**：检索于 2026-11-03，使用 `CSI feedback`、`massive MIMO CSI feedback`、
`multi-resolution CSI feedback`、`spatially separable attention CSI feedback`、`COST2100`、
`CSI feedback benchmark`、`reproducible CSI feedback`、`CSI feedback survey`、
`CSI feedback open source dataset`；元数据经 Crossref 核对 DOI、卷期与页码。
