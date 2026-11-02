# Idea: <short-name>

## 状态

- 分支：`idea/<short-name>`
- 状态：探索中 / 已验证 / 已合并 / 已放弃
- 负责人：
- 开始日期：

## 动机

描述 baseline 的空白或观察到的现象。引用具体结果（场景、CR、NMSE/rho）。

## 假设

一句话说明核心假设，例如：

> 在低压缩比下，频域与空域特征的联合建模可以减少量化误差。

## 方法

- 关键模块：
- 与 baseline 的差异：
- 参数量 / 计算量变化：
- 训练策略变化：

## 配置

列出实验配置路径：

```text
csifeedback/configs/ideas/<short-name>/<short-name>_indoor_cr4.yaml
...
```

## 结果

| 场景 | CR | baseline 最佳 NMSE | 本方法 NMSE | 提升 | 备注 |
|------|----|--------------------|-------------|------|------|
| indoor | 1/4 | | | | |
| indoor | 1/16 | | | | |

- 随机种子：
- 训练轮数：
- checkpoint 选择策略：
- 运行目录：

## 消融

| 变体 | NMSE | rho | 说明 |
|------|------|-----|------|
| full | | | |
| w/o module A | | | |

## 结论

- 是否达到预期：
- 主要收益来源：
- 失败模式 / 局限性：
- 后续建议：

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config csifeedback/configs/ideas/<short-name>/<short-name>_indoor_cr4.yaml
```
