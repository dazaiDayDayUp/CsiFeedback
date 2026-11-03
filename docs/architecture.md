# Baseline 模型架构

本文档详细说明三个 baseline（CLNet、STNet、CsiNet）的网络结构、数据流，以及与论文的对应关系。

> 这三个模型是公平对比的固定 baseline，网络结构与训练行为保持稳定。
> 除非修复明确的实现错误，否则不要修改它们。
> 新研究模型的接入方式见 [`research_workflow.md`](research_workflow.md)。

## 目录

- [统一约定](#统一约定)
- [CLNet](#clnet)
- [STNet](#stnet)
- [CsiNet](#csinet)
- [训练差异](#训练差异)
- [压缩比映射](#压缩比映射)

---

## 统一约定

三个模型都继承自 `csifeedback.models.base.CSIAutoencoder`，共享以下接口：

- `encoded_dim`：瓶颈维度（码字长度）。
- `forward(x)`：输入输出形状均为 `(N, 2, 32, 32)`，其中通道 0 为实部，通道 1 为虚部。
- `get_encoder_state()` / `get_decoder_state()`：分别返回 encoder/decoder 的状态字典。

输入数据已经过归一化，原始复数 CSI 的实部/虚部被映射到 `[0, 1]` 区间。因此指标计算（`csi_metrics.py`）会先用 `zero_point=0.5` 对预测和真值去中心化，再计算复数功率。

---

## CLNet

> 论文：*Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback*（Ji & Li，IEEE WCL 2021）

### 整体结构

```
输入 (N, 2, 32, 32)
    ↓
Encoder
    ↓
瓶颈 (N, 2048/reduction, 1)
    ↓
Decoder
    ↓
输出 (N, 2, 32, 32)
```

### Encoder

Encoder 由三条分支组成，最后通过 `Conv1d` 将 2048 维特征压缩到 `2048 / reduction` 维：

1. **多尺度卷积分支 `encoder1`**：
   - `Conv2d(2, 2, 3×3)` + BN + LeakyReLU
   - `Conv2d(2, 2, 1×9)` + BN + LeakyReLU
   - `Conv2d(2, 2, 9×1)` + BN
   - 后跟 **SpatialGate** 空间注意力

2. **通道注意力分支 `encoder2`**：
   - `Conv2d(2, 32, 1×1)` + BN
   - 后跟 **SELayer** 通道注意力（reduction=16）

3. **融合与压缩**：
   - 将 `encoder1` 输出（2 通道）与 `encoder2` 输出（32 通道）拼接，得到 34 通道
   - `Conv2d(34, 2, 1×1)` + BN + LeakyReLU
   - reshape 为 `(N, 2048, 1)`
   - `Conv1d(2048, 2048 // reduction, 1)` 完成压缩

### Decoder

Decoder 先将瓶颈向量恢复到 2048 维，再解码回 2×32×32：

1. `ConvTranspose1d(2048 // reduction, 2048, 1)`
2. reshape 为 `(N, 2, 32, 32)`
3. `Conv2d(2, 2, 5×5)` + BN + LeakyReLU
4. 两个 `CRBlock`
5. `hsigmoid` 输出

### 关键模块

- **CRBlock**：双分支残差块。
  - path1：`3×3` → `1×3` → `3×1`
  - path2：`1×5` → `5×1`
  - 拼接后 `1×1` 降维，再残差相加。

- **SpatialGate**：通道池化（max + mean）+ `3×3` 卷积 + Sigmoid，生成空间注意力图。

- **SELayer**：全局平均池化 → FC → ReLU → FC → Sigmoid，生成通道注意力权重。

- **hsigmoid**：`F.relu6(x + 3) / 6`，不是普通 Sigmoid。

### 初始化

Encoder 和 Decoder 内部使用 PyTorch 默认初始化；`CLNet` 最外层再次对所有 `Conv2d`/`Linear` 应用 `xavier_uniform_`，对所有 `BatchNorm2d` 设置 `weight=1, bias=0`。

---

## STNet

> 论文：*A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback*（Mourya & Amuru，arXiv 2022）

### 整体结构

```
输入 (N, 2, 32, 32)
    ↓
Encoder
    ↓
瓶颈 (N, encoded_dim)
    ↓
Decoder
    ↓
输出 (N, 2, 32, 32)
```

### Encoder

1. `Conv2d(2, 16, 1×1)` 升维
2. `Conv2d(16, 2, 5×5, padding=2)`
3. 保存残差 `X`
4. 经过 `depth` 个 **WTL** 块（默认 depth=1）
5. LayerNorm
6. `ConvTranspose2d(2, 2, 4×4, stride=2, padding=1)` 上采样
7. 与残差 `X` 相加：`x = X + conv4(x)`，其中 `conv4` 为 `Conv2d(2, 2, 4×4, stride=2, padding=1)` 下采样
8. LayerNorm
9. reshape 为 `(N, 2 * 32 * 32)` 后通过 `Linear` 投影到 `encoded_dim`

### Decoder

Decoder 的一个关键实现是：**`decoder_feature` 产生的 `out` 张量被复用到两个 WTL 循环中**。`out` 与两个循环中的 `x` 都是 `(N, 2, 32, 32)`，因此 `x + out` 是同形状逐元素相加，不涉及广播。

1. `Linear(encoded_dim, 2 * 32 * 32)` 恢复特征
2. reshape 为 `(N, 2, 32, 32)`
3. `decoder_feature`：`Conv2d(2, 2, 5×5, padding=2)` + BN + LeakyReLU + CRBlock，得到 `out`（形状 `(N, 2, 32, 32)`）
4. 第一个 WTL 循环：
   - `x = conv5(img)`
   - 对每个 WTL 块：`x = block(x + out)`
   - LayerNorm + `ConvTranspose2d`（32×32 → 64×64）+ `conv4`（64×64 → 32×32）
5. 第二个 WTL 循环：
   - 对每个 WTL 块：`x = block(x + out)`
   - LayerNorm + Sigmoid

> 注意：两个 WTL 循环都在 32×32 分辨率上进行，`out` 的复用是该 baseline 的固定行为。

### 关键模块

- **WTL（Window Transformer Layer）**：
  - `GroupAttention`：在 8×8 窗口内做多头自注意力。
  - `GlobalAttention`：对下采样后的键值做全局自注意力。
  - 每个 attention 后接 MLP（GELU 激活）。

- **CRBlock**：与 CLNet 的 CRBlock 类似，但 path1 使用 `1×9` + `9×1`（CLNet 为 `1×3` + `3×1`）。

- **hsigmoid**：同样使用 `F.relu6(x + 3) / 6`。

### 训练差异

STNet 使用两个独立的 Adam 优化器分别优化 encoder 和 decoder，betas 为 `(0.5, 0.999)`。

---

## CsiNet

> 论文：*Deep Learning for Massive MIMO CSI Feedback*（Wen 等人，IEEE WCL 2018）

### 整体结构

```
输入 (N, 2, 32, 32)
    ↓
Encoder
    ↓
瓶颈 (N, encoded_dim)
    ↓
Decoder
    ↓
输出 (N, 2, 32, 32)
```

### Encoder

CsiNet 的 Encoder 非常简洁：

1. `Conv2d(2, 2, 3×3, padding=1, bias=True)` + BN + LeakyReLU
2. Flatten
3. `Linear(2 * 32 * 32, encoded_dim)`

> 注意：encoder 的卷积使用 `bias=True`。

### Decoder

1. `Linear(encoded_dim, 2 * 32 * 32)`
2. reshape 为 `(N, 2, 32, 32)`
3. 两个 `ResidualBlock`
4. `Conv2d(2, 2, 3×3, padding=1)`
5. Sigmoid

### ResidualBlock

```
Conv2d(2, 8, 3×3, bias=False) → BN → LeakyReLU
Conv2d(8, 16, 3×3, bias=False) → BN → LeakyReLU
Conv2d(16, 2, 3×3, bias=False) → BN
残差相加 → LeakyReLU
```

> 注意：ResidualBlock 中所有卷积使用 `bias=False`。

### 训练差异

CsiNet 使用单个 Adam 优化器，并按照验证损失保存最优模型（`best_val.pth`）。

---

## 训练差异

| 模型 | 优化器 | 学习率调度 | 最优保存依据 |
|------|--------|-----------|-------------|
| CLNet | 单 Adam，betas=(0.9, 0.999) | cosine warmup（默认，lr=0.002）或 const（lr=1e-3） | test rho / test NMSE |
| STNet | 双 Adam（enc/dec 分离），betas=(0.5, 0.999) | const | test rho / test NMSE |
| CsiNet | 单 Adam，betas=(0.9, 0.999) | const | val loss |

统一入口 `scripts/train.py` 会根据 `model.name` 自动选择对应 trainer；配置中的 `optimizer.betas` 和 `scheduler.name` 需要与模型匹配，具体可参考 `csifeedback/configs/baselines/` 下的示例 YAML。`optimizer.separate_enc_dec` 目前只是保留字段，trainer 选择由 `model.name` 决定。

CLNet 的 cosine warmup：warmup 阶段学习率从 0 线性增长到初始 lr，之后进入余弦退火到 `eta_min`。

---

## 压缩比映射

| 压缩比 | CLNet (`reduction`) | STNet (`encoded_dim`) | CsiNet (`encoded_dim`) |
|--------|---------------------|-----------------------|------------------------|
| 1/4  | 4  | 512 | 512 |
| 1/8  | 8  | 256 | 256 |
| 1/16 | 16 | 128 | 128 |
| 1/32 | 32 | 64  | 64  |
| 1/64 | 64 | 32  | 32  |

输入 CSI 特征图总维度为 `2 × 32 × 32 = 2048`，因此瓶颈维度与压缩比直接对应。

---

## 扩展接口

新模型通过 `csifeedback/models/registry.py` 的 `@register_model` 注册，不需要修改
已有模型文件。所有模型共享：

- `CSIAutoencoder`：要求实现 `encoded_dim`、`get_encoder_state()`、
  `get_decoder_state()` 和 `forward(x)`；
- 统一的 `forward` 输入输出形状 `(N, 2, 32, 32)`；
- 统一的数据加载与 NMSE/rho 指标。

如果新模型只需要标准 MSE + 单优化器，`DefaultTrainer` 会根据 YAML 自动构建
Adam / AdamW / SGD 和 const / cosine 调度器，无需新增 trainer。详细步骤见
[`research_workflow.md`](research_workflow.md)。
