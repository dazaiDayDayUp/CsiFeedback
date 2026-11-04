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

## 方法

- 编码器：卷积主干下采样到 4×4；路径头输出 `K` 条径的 `(angle, delay, gain_real, gain_imag)`；
  残差头输出 `4×4` 空间特征，展平后作为漫散射码字。
- 码字：`K*4` 维路径参数 + `encoded_dim - K*4` 维残差，总维度严格等于 `2048/CR`。
- 解码器：每条径在连续角度、时延上计算一维 Dirichlet 核，做外积得到 32×32 复响应，
  乘复增益后求和；残差分支用转置卷积上采样到 32×32；两者相加后激活。
- 辅助损失：对“仅路径合成”的输出额外算 MSE，权重 0.2，防止残差分支独占重建。
- 峰对齐损失：把 K 条路径按增益幅度排序后，与输入角-时延域 top-K 幅度峰逐条对齐；
  前 150 epoch 权重 0.005，再退火 250 epoch 到 0。它解决连续频率在随机初始化时
  梯度过小、路径分支停在零附近的平台问题。

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

待填。

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
