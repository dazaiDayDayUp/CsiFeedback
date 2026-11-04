# Idea: path_geometry_feedback

## 状态

- 分支：`idea/path_geometry_feedback`
- 状态：实现 / smoke test
- 开始日期：2026-11-04

## 动机

COST2100 的角-时延域 CSI 在 indoor 场景高度集中：实测 rank-1 占总能量约 89.4%，
rank-2 占 97.4%；outdoor 更弥散（rank-1 约 59.3%，rank-2 约 79.0%）。
现有 baseline 把 32×32 网格当图像处理，没有显式利用“CSI 是少量离格多径叠加”
这一物理结构。

## 假设

> 把码字显式组织为若干条径的连续角度、时延和复增益，并用离格 Dirichlet 核
> 合成 rank-one 多径，再用一个小残差分支补偿漫散射，可以在相同码字预算下
> 比纯网格卷积/注意力获得更好的低压缩比重建。

## 误差分析（STNet CR=1/4）

对 STNet `best_nmse.pth` 做逐样本分析（20000 test）：

- indoor：worst 10% 样本的 rank2 能量 0.950（best 10% 为 0.997），
  delay entropy 1.674（best 10% 为 0.998）；rank2 与 NMSE 的 Spearman 为 -0.501。
- outdoor：worst 10% 的 rank2 能量 0.721（best 10% 为 0.920），
  delay entropy 2.861（best 10% 为 1.781）；rank2 相关 -0.429。

结论：baseline 的难点是**弥散、低秩性弱、时延扩展大**的信道，而不是强 specular
样本。因此残差/漫散射分支是关键；本 idea 把残差解码器宽度从 64 加宽到 128。

## 方法

- 编码器：卷积主干先在 16×16 特征上做多径峰值检测，上采样到 32×32，输出
  一张峰值热图和一张复增益图；直接取热图 top-K 峰（不做 NMS，因为同一 delay
  上相邻角度可能是不同物理径），再在每个峰周围做 3×3 soft-argmax，得到连续
  `(angle, delay, gain_real, gain_imag)`。
- 码字：`K*4` 维路径参数 + `encoded_dim - K*4` 维残差，总维度严格等于 `2048/CR`。
- 解码器：每条径在连续角度、时延上计算一维 Dirichlet 核，做外积得到 32×32 复响应，
  乘复增益后求和；残差分支用转置卷积上采样到 32×32；两者相加后激活。
- 物理辅助损失：
  - 热图损失：目标为输入角-时延幅度 top-K 峰的高斯热图；
  - 增益图损失：在目标峰位置回归复增益；
  - 路径 token 匹配损失：把提取出的 K 条路径与 top-K 峰贪心最近邻匹配，
    直接监督角度、时延和增益。
- 两阶段训练：前 150 epoch 只训练路径分支（残差分支不参与损失），
  之后联合训练最终重建与路径辅助重建，让 specular 几何先学会、diffuse 残差再补足。

## 与近邻工作的区别

- HS-PINNnet 输出多径参数，但主要做参数化重建；这里显式保留漫散射残差码字，
  并把码字预算拆成“几何 + 残差”两部分。
- LRP 用秩一感知/合成；这里不限制秩一算子，而是用离格 Dirichlet 核建模
  截断 DFT 的泄漏，并保留非线性残差分支。
- PM-VQ 用平均功率图先验；这里是逐样本的显式路径几何，不依赖统计先验。

## 配置

```text
csifeedback/configs/ideas/path_geometry_feedback/path_geometry_feedback_indoor_cr4.yaml
```

## 结果

### 2026-11-04 两阶段 screening（indoor CR4，300 epoch，提前停止）

运行目录：`experiments/path_geometry_feedback/indoor_cr4/20261104_232336`。

| epoch | 阶段 | test NMSE (dB) | rho |
|------:|------|---------------:|----:|
| 10 | path only | -5.00 | 0.858 |
| 50 | path only | -6.01 | 0.891 |
| 90 | path only | -5.75 | 0.882 |
| 100 | stage 2 开始 | -3.25 | 0.746 |
| 110 | stage 2 | -6.01 | 0.888 |
| 120 | stage 2 | -4.08 | 0.825 |

诊断：热图 top-K 与目标峰重合 95%；单样本纯路径 -5.6 dB，接近该样本
top-8 系数 oracle -6.9 dB，说明路径分支已接近 K=8 上界。瓶颈是弥散残差，
stage 2 的残差分支没有接上，训练不稳定。同期 STNet epoch 300 为 -26.67 dB，
当前配方明显不可用。

### joint CPU 诊断（512 样本，60 epoch）

- full NMSE **-5.45 dB**，path-only -3.86 dB；
- 对比两阶段同规模 full -3.68 dB：joint 更好，残差分支确实能贡献约 1.6 dB；
- 因此 GPU joint screening 优先于继续调两阶段。

### residual_main CPU 诊断（512 样本，60 epoch）

- full NMSE **-10.50 dB**，path-only -2.73 dB；
- 对比 joint full -5.45 dB、两阶段 -3.68 dB：**最终 MSE 主导、路径只做辅助**
  明显更好。强迫路径分支重建（path_loss_weight=0.2）会伤害最终性能。
- 因此 GPU screening 改为 residual_main 配置：
  `path_pretrain_epochs=0`、`path_loss_weight=0`、aux 权重 0.05、退火 20/30。

下一步（fallback 阶梯）：

1. **residual_main**：最终 MSE 主导，路径作为物理辅助头（GPU 优先，已排队）。
2. **joint**：`path_pretrain_epochs=0`，残差从 epoch 1 参与（已被 residual_main 取代）。
2. **residual_main**：`path_loss_weight=0`，最终 MSE 主导，路径分支只作为物理辅助
   头（热图/增益/token 匹配），配置见
   `ablations/pgf_indoor_cr4_residual_main.yaml`。
3. 若两级都失败，记录负结果并转向 `angle_delay_equivariant` 或其他新 idea。

## 消融计划

| 变体 | 说明 |
|------|------|
| full | 路径 + 残差 |
| path only | 去掉残差分支 |
| grid kernel | 用最近网格点代替离格 Dirichlet 核 |
| no aux loss | 去掉路径辅助损失 |
| K sweep | K = 2 / 4 / 8 / 16 |

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/path_geometry_feedback/path_geometry_feedback_indoor_cr4.yaml
```
