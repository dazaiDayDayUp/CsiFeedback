r"""CylinderCodec：几何正确的角度-时延 CSI 反馈自编码器。

物理出发点
----------
COST2100 的 ``(2, 32, 32)`` 表示是角时延域图像：第二维是 32 个阵元的空间 DFT
（角度 bin），第三维是 32 个子载波 DFT 的前若干个时延 tap。这个表示流形不是平面：

1. **角度轴是圆周**。均匀线阵的阵列流形以 1 为周期（空间频率混叠），32 点 DFT
   的角度 bin 首尾相接：bin 31 与 bin 0 是相邻波束方向。零填充会在边界制造
   不存在的"墙"，把本该连续的能量切断。实测 COST2100 indoor 在 bin 30/31 的
   能量回升（0.026/0.037），正是回绕证据。
2. **时延轴是半直线**。时延非负，tap 0 是首达径，左侧没有负时延；右侧是
   32 tap 的截断。因此左侧用零（吸收）边界，而不是镜像/周期边界。

据此，本模型的卷积算子全部使用「角度圆周 + 时延半直线」的边界条件
（``CylindricalConv2d``），这是对表示流形几何的硬约束，不引入额外参数。

叠加 ``MetricMSELoss``（逐样本归一化的 NMSE，即部署指标本身），使训练目标与
报告指标完全一致，而不是功率加权的逐元素 MSE。
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


__all__ = ["CylinderCodec", "CylindricalConv2d", "create_cylinder_codec"]

_ANGLE = 32
_DELAY = 32


def cylindrical_pad(x: torch.Tensor, pad: int, circular: bool = True) -> torch.Tensor:
    """只对角度轴填充：圆周（几何正确）或零（消融对照）。"""
    if pad == 0:
        return x
    return F.pad(x, (0, 0, pad, pad), mode="circular" if circular else "constant")


class CylindricalConv2d(nn.Conv2d):
    """角度圆周（可切换为零填充）、时延零填充的卷积。"""

    def __init__(self, *args, circular: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.circular = circular

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        x = cylindrical_pad(x, pad_h, self.circular) if pad_h else x
        if pad_w:
            x = F.pad(x, (pad_w, pad_w, 0, 0))
        return F.conv2d(
            x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups
        )


class LayerNorm2d(nn.Module):
    """通道维 LayerNorm（NHWC 语义），比 BatchNorm 更适合小 batch 与生成式解码。"""

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


class SE(nn.Module):
    """通道注意力。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class CylBlock(nn.Module):
    """角度-时延各向异性残差块。

    先在角度轴上做 5×1 圆周卷积（波束方向的连续性），再在时延轴上做 1×5 卷积
    （功率时延谱的局部性），两个方向的边界条件由 ``CylindricalConv2d`` 保证。
    """

    def __init__(self, channels: int, dilation: int = 1, anisotropic: bool = True,
                 circular: bool = True):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        if anisotropic:
            self.conv_a = CylindricalConv2d(
                channels, channels, (5, 1), padding=(2 * dilation, 0),
                dilation=(dilation, 1), circular=circular,
            )
            self.conv_d = CylindricalConv2d(
                channels, channels, (1, 5), padding=(0, 2 * dilation),
                dilation=(1, dilation), circular=circular,
            )
        else:
            self.conv_a = CylindricalConv2d(
                channels, channels, 5, padding=2 * dilation, dilation=dilation,
                circular=circular,
            )
            self.conv_d = None
        self.se = SE(channels)
        self.norm2 = LayerNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.gelu(self.conv_a(h))
        if self.conv_d is not None:
            h = F.gelu(self.conv_d(h))
        return x + self.norm2(self.se(h))


class Encoder(nn.Module):
    def __init__(self, encoded_dim: int, channels: int = 64, depth: int = 8,
                 anisotropic: bool = True, circular: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            CylindricalConv2d(2, channels, 3, padding=1, circular=circular),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        dilations: List[int] = [1, 1, 2, 2, 4, 4, 8, 8][:depth]
        while len(dilations) < depth:
            dilations.append(8)
        self.blocks = nn.ModuleList(
            [CylBlock(channels, d, anisotropic, circular) for d in dilations]
        )
        self.head = nn.Sequential(
            LayerNorm2d(channels),
            CylindricalConv2d(channels, channels // 4, 1, circular=circular),
            nn.GELU(),
            CylindricalConv2d(channels // 4, 2, 1, circular=circular),
        )
        self.fc = nn.Linear(2 * _ANGLE * _DELAY, encoded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.head(h)
        return self.fc(h.flatten(1))


class Decoder(nn.Module):
    def __init__(self, encoded_dim: int, channels: int = 64, depth: int = 8,
                 output: str = "tanh", anisotropic: bool = True,
                 circular: bool = True):
        super().__init__()
        if output not in {"tanh", "linear"}:
            raise ValueError(f"output 必须是 tanh/linear，得到 {output!r}")
        self.output = output
        self.fc = nn.Linear(encoded_dim, 2 * _ANGLE * _DELAY)
        self.stem = nn.Sequential(
            CylindricalConv2d(2, channels, 3, padding=1, circular=circular),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        dilations: List[int] = [1, 1, 2, 2, 4, 4, 8, 8][:depth]
        while len(dilations) < depth:
            dilations.append(8)
        self.blocks = nn.ModuleList(
            [CylBlock(channels, d, anisotropic, circular) for d in dilations]
        )
        self.head = nn.Sequential(
            LayerNorm2d(channels),
            CylindricalConv2d(channels, channels // 4, 1, circular=circular),
            nn.GELU(),
            CylindricalConv2d(channels // 4, 2, 1, circular=circular),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 2, _ANGLE, _DELAY)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.head(h)
        if self.output == "tanh":
            # CSI 幅度归一化到 [0,1]，用有界输出把值域作为硬约束，同时让初始输出为 0.5
            return 0.5 + 0.5 * torch.tanh(h)
        return h + 0.5


class CylinderCodec(CSIAutoencoder):
    """角度圆周 + 时延半直线几何约束下的 CSI 自编码器。"""

    def __init__(
        self,
        encoded_dim: int = 512,
        channels: int = 64,
        depth: int = 8,
        output: str = "tanh",
        anisotropic: bool = True,
        circular: bool = True,
    ):
        super().__init__()
        self._encoded_dim = encoded_dim
        self.encoder = Encoder(encoded_dim, channels, depth, anisotropic, circular)
        self.decoder = Decoder(
            encoded_dim, channels, depth, output, anisotropic, circular
        )

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> Dict:
        return self.encoder.state_dict()

    def get_decoder_state(self) -> Dict:
        return self.decoder.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@register_model("cylinder_codec")
def create_cylinder_codec(
    encoded_dim: int = 512,
    channels: int = 64,
    depth: int = 8,
    output: str = "tanh",
    anisotropic: bool = True,
    circular: bool = True,
    **kwargs,
) -> CylinderCodec:
    """CylinderCodec 工厂函数。"""
    return CylinderCodec(
        encoded_dim=encoded_dim,
        channels=channels,
        depth=depth,
        output=output,
        anisotropic=anisotropic,
        circular=circular,
    )
