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

- 编码器：卷积主干先在 16×16 特征上做多径峰值检测，上采样到 32×32，输出
  一张峰值热图和一张复增益图；取热图 top-K 峰，在每个峰周围做 3×3 soft-argmax，
  得到连续 `(angle, delay, gain_real, gain_imag)`。
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
