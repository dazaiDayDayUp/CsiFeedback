# Idea: angle_delay_equivariant

## 状态

- 分支：`idea/angle_delay_equivariant`
- 状态：实现 / smoke test
- 开始日期：2026-11-04

## 动机

COST2100 的角-时延域来自 32 点角度 DFT 与 1024 点时延 DFT 的前 32 个 tap。
角度维天然周期，时延维是截断边界；STNet/CLNet 都用零 padding 的二维卷积，
把角度维当成普通图像边界，破坏了角度 DFT 的圆周平移对称性。

## 假设

> 用角度圆周 padding + 时延零 padding 的卷积，并在注意力里加入
> 角度周期余弦/时延高斯相对偏置，可以给模型正确的物理归纳偏置，
> 在相同码字预算下提升重建精度，并增强对角度/时延平移的鲁棒性。

## 方法

- `CircularConv2d`：角度维 `circular` padding，时延维零 padding。
- `PhysicsAttention2d`：8×8 特征上的多头自注意力，attention logits 加
  `alpha * cos(2π Δangle) + beta * exp(-Δdelay²/2σ²)`；alpha/beta 用
  `2*tanh(raw)` 有界参数化。base 300 epoch 诊断发现 encoder alpha 增长到 9.64，
  导致 attention softmax 饱和、测试曲线反复跌落；有界化用于后续配方。
- 编码器：2→64→96→128，8×8 注意力，1×1 降到 16 通道，线性到码字。
- 解码器：线性回 8×8×16，8×8 注意力，两次转置卷积回 32×32，最后圆周卷积 + sigmoid。

## 与近邻工作的区别

- STNet 的窗口/全局注意力只建模局部窗口和下采样全局 token，没有显式的
  角度周期性或时延截断先验。
- 本 idea 不是“加一个注意力模块”，而是把角-时延域的两种边界条件写进卷积
  和注意力偏置；额外的评价维度是角度/时延平移鲁棒性。

## 配置

```text
csifeedback/configs/ideas/angle_delay_equivariant/angle_delay_equivariant_indoor_cr4.yaml
```

## 结果

待填。

## 消融计划

| 变体 | 说明 |
|------|------|
| full | 圆周卷积 + 物理偏置 |
| zero padding | 全部换回零 padding |
| no bias | alpha=beta=0，保留圆周卷积 |
| attention off | 去掉 PhysicsAttention |
| lr5e4 | 学习率 5e-4 的配方消融（base 300 epoch 有振荡） |
| cosine | cosine warmup 调度替代 const |

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/angle_delay_equivariant/angle_delay_equivariant_indoor_cr4.yaml
```
