# 原始代码归档

本目录保存了从三个独立 GitHub 仓库直接下载的原始实现，仅作为历史参考，**不再维护**。

- `CLNet/`：Complex Input Lightweight Neural Network（Ji & Li，IEEE WCL 2021）的原始 PyTorch 实现。
- `STNet/`：A Spatially Separable Attention Mechanism For Massive MIMO CSI Feedback（Mourya & Amuru，arXiv 2022）的原始实现。其中 `stnet.py` 存在作用域 bug，请使用 `stnet_fixed.py`。
- `CsiNet/`：Wen 等人（IEEE WCL 2018）的原始 Keras/TensorFlow 实现，以及本机 TensorFlow 无法安装时的 PyTorch 重写版本。

## 说明

1. 原始 Keras/TensorFlow 脚本依赖 TensorFlow，在当前环境（Python 3.12、无 TensorFlow）下无法运行。
2. actively maintained 的代码已迁移到顶层 `csifeedback/` 包中：
   - `csifeedback/models/clnet.py`
   - `csifeedback/models/stnet.py`
   - `csifeedback/models/csinet.py`
3. 归档时只保留原始源码与 README；训练日志、 checkpoints、Jupyter checkpoints、Python 缓存等中间产物已清理，避免干扰。

如需了解新的使用方式，请参阅顶层 `README.md` 和 `docs/migration.md`。
