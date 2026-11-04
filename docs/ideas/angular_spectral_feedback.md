# Idea: angular_spectral_feedback

## 状态

- 分支：`idea/angular_spectral_feedback`
- 状态：实现 / smoke test
- 开始日期：2026-11-05

## 动机

角-时延域的两维物理含义不同：角度维是 32 点 DFT，天然周期；时延维是
1024 点 DFT 截断到前 32 个 tap，是截断边界。STNet 的窗口/全局注意力
没有显式区分这两种边界；PGF 用显式路径分解，但训练复杂。

## 假设

> 在瓶颈特征上用“角度维 FFT 低阶谱卷积 + 时延维局部卷积”的谱-空间块，
> 可以同时获得角域全局平面波混合和时延域局部建模，比纯窗口注意力更适合
> 角度/时延结构，且比显式路径分解更容易训练。

## 方法

- `AngularSpectralConv`：对角度维做 `rfft`，保留前 `modes` 个低阶模式，
  用可学习复权重做谱乘，再 `irfft`；随后沿时延维做 3-tap 局部卷积。
- `SpectralBlock`：GroupNorm → 角谱-时延卷积 → GELU → 1×1 → 残差。
- 编码器：2→64→96→128，8×8 上一个谱块，1×1 降到 16，线性到码字。
- 解码器：线性回 8×8×16，1×1 升到 64，两个谱块，两次转置卷积回 32×32，sigmoid。

## 与近邻工作的区别

- STNet：窗口/下采样全局注意力，没有角域周期和时延截断的显式区分。
- PGF：显式多径参数 + Dirichlet 核，训练复杂；本 idea 不显式输出路径，
  而是在特征域做角谱混合。
- ADEF：圆 padding + 物理相对偏置注意力；本 idea 用 FFT 谱乘，不做 attention。

## 配置

```text
csifeedback/configs/ideas/angular_spectral_feedback/angular_spectral_feedback_indoor_cr4.yaml
```

## 结果

待填。

## 消融计划

| 变体 | 说明 |
|------|------|
| full | 角谱 + 时延卷积 |
| angle conv only | 去掉角度谱，改普通 2D 卷积 |
| delay global | 时延维也做 FFT 混合 |
| modes sweep | modes = 2 / 5 / 8 |

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/angular_spectral_feedback/angular_spectral_feedback_indoor_cr4.yaml
```
