r"""AxisCodec：算力对等的角度-时延 CSI 反馈网络。

设计动机
--------
baseline（CLNet/STNet/CsiNet）的计算量都在 0.01 GFLOPs/样本量级：它们在很窄的
通道（2 通道）上沿空间轴做混合。若在 32×32 全分辨率上用 64 通道卷积，FLOPs 会
高两个数量级，无法做公平的复杂度对比。

本模型因此沿两条空间轴做**可分离的全局混合**，通道数很小：

- **角度轴**：阵列流形以 1 为周期，角度 bin 首尾相接（圆群）。因此角度混合器
  用**循环卷积**（在 DFT 域乘一个学到的 32 点滤波器），参数量 32 而不是 1024，
  且天然满足周期边界。
- **时延轴**：时延非负、能量沿 tap 单调衰减，**没有平移不变性**。因此时延混合器
  用稠密线性映射（32×32），让每个 tap 拥有独立的读出头。

两条轴的对称性不同，算子就不同——这是把表示流形的几何写进结构，而不是相同的
卷积核简单堆叠。

训练目标默认用 ``MetricMSELoss``（与报告指标只差 log10）。
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


__all__ = ["AxisCodec", "AxisBlock", "CircularMix", "create_axis_codec"]

_ANGLE = 32
_DELAY = 32


class LayerNorm2d(nn.Module):
    """通道维 LayerNorm：逐位置归一化，使弱能量的时延尾部与头部获得同等学习信号。"""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def make_norm(kind: str, channels: int) -> nn.Module:
    """归一化层工厂。

    ``ln``：逐位置通道 LayerNorm（抹掉每个空间位置的绝对幅度）；
    ``bn``：BatchNorm2d（逐通道统计，保留空间幅度差异 —— 弱时延 tap 依赖这一信息）。
    """
    kind = kind.lower()
    if kind == "ln":
        return LayerNorm2d(channels)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"norm 必须是 ln/bn，得到 {kind!r}")


class CircularMix(nn.Module):
    """角度轴循环卷积：DFT 域乘逐通道学到的滤波器（每通道 32 个自由参数）。"""

    def __init__(self, channels: int, length: int = _ANGLE):
        super().__init__()
        self.length = length
        n_freq = length // 2 + 1
        # 恒等滤波器：频域权重全 1（irfft(rfft(x)*1) == x），保证初始为恒等映射
        self.weight_re = nn.Parameter(torch.ones(1, channels, n_freq, 1))
        self.weight_im = nn.Parameter(torch.zeros(1, channels, n_freq, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(x, dim=2)
        w = torch.complex(self.weight_re, self.weight_im).to(spec.dtype)
        return torch.fft.irfft(spec * w, n=self.length, dim=2)


class DenseMix(nn.Module):
    """沿指定空间轴的稠密线性混合（axis=2 角度，axis=3 时延）。"""

    def __init__(self, length: int = _ANGLE, axis: int = 2):
        super().__init__()
        self.axis = axis
        self.fc = nn.Linear(length, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == 3:
            return self.fc(x)
        return self.fc(x.transpose(2, 3)).transpose(2, 3)


class AxisBlock(nn.Module):
    """先角度轴（周期）混合，再时延轴（无对称性）混合，最后通道 MLP。"""

    def __init__(self, channels: int, angle_mode: str = "circular", mlp_ratio: int = 2,
                 norm: str = "ln"):
        super().__init__()
        if angle_mode not in {"circular", "dense"}:
            raise ValueError(f"angle_mode 必须是 circular/dense，得到 {angle_mode!r}")
        self.angle_mode = angle_mode
        self.norm = make_norm(norm, channels)
        self.angle = (
            CircularMix(channels, _ANGLE) if angle_mode == "circular"
            else DenseMix(_ANGLE, axis=2)
        )
        self.delay = DenseMix(_DELAY, axis=3)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )
        self.norm2 = make_norm(norm, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.angle(h)
        h = F.gelu(h)
        h = self.delay(h)
        h = F.gelu(h)
        h = self.mlp(h)
        return x + self.norm2(h)


class Encoder(nn.Module):
    def __init__(self, encoded_dim: int, channels: int = 8, depth: int = 8,
                 angle_mode: str = "circular", pca_basis=None, norm: str = "ln"):
        super().__init__()
        if pca_basis is None:
            self.pca_basis = None
        else:
            # persistent=False：不写入 state_dict，但会随 .to(device) 迁移
            self.register_buffer("pca_basis", pca_basis, persistent=False)
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), make_norm(norm, channels), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [AxisBlock(channels, angle_mode, norm=norm) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            make_norm(norm, channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
        )
        self.fc = nn.Linear(2 * _ANGLE * _DELAY, encoded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        z = self.fc(self.head(h).flatten(1))
        if self.pca_basis is not None:
            # 固定 PCA 投影 + 学习的残差编码（fc 零初始化 ⇒ 初始即 PCA 系数）
            z = z + F.linear((x - 0.5).flatten(1), self.pca_basis)
        return z


class Decoder(nn.Module):
    def __init__(self, encoded_dim: int, channels: int = 8, depth: int = 8,
                 angle_mode: str = "circular", output: str = "linear",
                 bypass: bool = True, pca_basis=None, norm: str = "ln"):
        super().__init__()
        if pca_basis is None:
            self.pca_basis = None
        else:
            self.register_buffer("pca_basis", pca_basis, persistent=False)
        if output not in {"tanh", "linear"}:
            raise ValueError(f"output 必须是 tanh/linear，得到 {output!r}")
        self.output = output
        self.fc = nn.Linear(encoded_dim, 2 * _ANGLE * _DELAY)
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), make_norm(norm, channels), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [AxisBlock(channels, angle_mode, norm=norm) for _ in range(depth)]
        )
        # 线性旁路：主干中的 LayerNorm2d 逐位置归一化会抹掉弱能量位置（时延尾部）
        # 的绝对幅度，旁路直接把 stem 特征线性映射到输出，保留幅度信息。
        self.bypass = nn.Conv2d(channels, 2, 1) if bypass else None
        self.head = nn.Sequential(
            make_norm(norm, channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 线性基座：PCA 模式下是固定合成基（+ 零初始化的可学习修正），否则是可学习线性映射。
        base = self.fc(z).view(-1, 2, _ANGLE, _DELAY)
        if self.pca_basis is not None:
            base = base + F.linear(z, self.pca_basis.t()).view(-1, 2, _ANGLE, _DELAY)
        h = self.stem(base)
        skip = self.bypass(h) if self.bypass is not None else None
        for blk in self.blocks:
            h = blk(h)
        res = self.head(h)
        if skip is not None:
            res = res + skip
        out = base + res
        if self.output == "tanh":
            return 0.5 + 0.5 * torch.tanh(out)
        return 0.5 + out


class AxisCodec(CSIAutoencoder):
    """算力对等的角度-时延自编码器。"""

    def __init__(
        self,
        encoded_dim: int = 512,
        channels: int = 8,
        depth: int = 8,
        angle_mode: str = "circular",
        output: str = "linear",
        bypass: bool = True,
        pca_path: str = "",
        norm: str = "ln",
    ):
        super().__init__()
        self._encoded_dim = encoded_dim
        basis = None
        if pca_path:
            payload = torch.load(pca_path, map_location="cpu", weights_only=False)
            basis = payload["basis"].float()
            if tuple(basis.shape) != (encoded_dim, 2 * _ANGLE * _DELAY):
                raise ValueError(
                    f"PCA 基形状 {tuple(basis.shape)} 与 encoded_dim={encoded_dim} 不匹配"
                )
            self.register_buffer("pca_basis", basis, persistent=False)
        self.encoder = Encoder(encoded_dim, channels, depth, angle_mode, basis, norm)
        self.decoder = Decoder(
            encoded_dim, channels, depth, angle_mode, output, bypass, basis, norm
        )
        if basis is not None:
            self._zero_init_residuals()

    def _zero_init_residuals(self) -> None:
        """PCA 初始化：残差分支零初始化，使初始前向严格等于 PCA 自编码器。"""
        for module in (self.encoder.fc, self.decoder.fc, self.decoder.bypass):
            if hasattr(module, "weight"):
                nn.init.zeros_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        last = self.decoder.head[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> Dict:
        return self.encoder.state_dict()

    def get_decoder_state(self) -> Dict:
        return self.decoder.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@register_model("axis_codec")
def create_axis_codec(
    encoded_dim: int = 512,
    channels: int = 8,
    depth: int = 8,
    angle_mode: str = "circular",
    output: str = "linear",
    bypass: bool = True,
    pca_path: str = "",
    norm: str = "ln",
    **kwargs,
) -> AxisCodec:
    """AxisCodec 工厂函数。"""
    return AxisCodec(
        encoded_dim=encoded_dim,
        channels=channels,
        depth=depth,
        angle_mode=angle_mode,
        output=output,
        bypass=bypass,
        pca_path=pca_path,
        norm=norm,
    )
