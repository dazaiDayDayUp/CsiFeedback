r"""Angle-Delay Equivariant Feedback：尊重角-时延域物理对称性的 CSI 自编码器。

物理先验：
- 角域是 32 点 DFT，天然具有圆周平移对称性；零 padding 会破坏它。
- 时延域是 1024 点 DFT 截断到前 32 个 tap，边界是截断而非周期。
因此编码器使用“角度圆周 padding + 时延零 padding”的卷积，并在 8×8 特征上
加入物理相对偏置注意力：角度差用周期余弦，时延差用高斯局部核，让注意力
显式偏向同一条多径/相邻时延的 token。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


__all__ = ["AngleDelayEquivariant", "create_angle_delay_equivariant"]


class CircularConv2d(nn.Module):
    """角度维圆周 padding、时延维零 padding 的二维卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            # 先补时延维（width），再补角度维（height，圆周）。
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="constant", value=0.0)
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="circular")
        return self.conv(x)


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__(
            CircularConv2d(
                in_channels, out_channels, kernel_size, stride=stride
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class PhysicsAttention2d(nn.Module):
    """带角度周期/时延局部相对偏置的自注意力块。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # 有界物理偏置：避免 alpha/beta 无约束增长导致 attention softmax 饱和。
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.beta_raw = nn.Parameter(torch.zeros(1))
        self.sigma = sigma
        self._angle_bias: Optional[torch.Tensor] = None
        self._delay_bias: Optional[torch.Tensor] = None
        self._cached_shape = None
        self._cached_device = None

    def _build_bias(self, height: int, width: int, device: torch.device) -> None:
        angle = torch.arange(
            height, device=device, dtype=torch.float32
        ).view(-1, 1) / height
        delay = torch.arange(
            width, device=device, dtype=torch.float32
        ).view(-1, 1) / width
        angle = angle.expand(height, width).reshape(-1, 1)
        delay = delay.expand(height, width).reshape(-1, 1)
        angle_diff = angle - angle.T
        delay_diff = delay - delay.T
        self._angle_bias = torch.cos(2.0 * math.pi * angle_diff)
        self._delay_bias = torch.exp(
            -(delay_diff ** 2) / (2.0 * self.sigma ** 2)
        )
        self._cached_shape = (height, width)
        self._cached_device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        if self._cached_shape != (h, w) or self._cached_device != x.device:
            self._build_bias(h, w, x.device)

        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        alpha = 2.0 * torch.tanh(self.alpha_raw)
        beta = 2.0 * torch.tanh(self.beta_raw)
        bias = alpha * self._angle_bias + beta * self._delay_bias
        attn, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=bias,
            need_weights=False,
        )
        tokens = tokens + attn
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(n, c, h, w)


class Encoder(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        base_channels: int = 64,
        num_heads: int = 4,
        sigma: float = 1.0,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels + 32
        c3 = base_channels + 64
        self.stem = ConvBlock(2, c1, 3)
        self.down1 = ConvBlock(c1, c2, 3, stride=2)  # 16×16
        self.down2 = ConvBlock(c2, c3, 3, stride=2)  # 8×8
        self.attn = PhysicsAttention2d(
            c3, num_heads=num_heads, sigma=sigma
        )
        self.reduce = nn.Conv2d(c3, 16, kernel_size=1)
        self.fc = nn.Linear(16 * 8 * 8, encoded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.attn(x)
        x = self.reduce(x)
        return self.fc(x.flatten(1))


class Decoder(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        base_channels: int = 64,
        num_heads: int = 4,
        sigma: float = 1.0,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels + 32
        c3 = base_channels + 64
        self.fc = nn.Linear(encoded_dim, 16 * 8 * 8)
        self.up0 = ConvBlock(16, c3, 3)
        self.attn = PhysicsAttention2d(
            c3, num_heads=num_heads, sigma=sigma
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.final = CircularConv2d(c1, 2, 3)
        self.act = nn.Sigmoid()

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        x = self.fc(code).view(-1, 16, 8, 8)
        x = self.up0(x)
        x = self.attn(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return self.act(x)


class AngleDelayEquivariant(CSIAutoencoder):
    """角度-时延对称/等变 CSI 反馈自编码器。"""

    def __init__(
        self,
        encoded_dim: int = 512,
        base_channels: int = 64,
        num_heads: int = 4,
        sigma: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self._encoded_dim = encoded_dim
        self.encoder = Encoder(
            encoded_dim=encoded_dim,
            base_channels=base_channels,
            num_heads=num_heads,
            sigma=sigma,
        )
        self.decoder = Decoder(
            encoded_dim=encoded_dim,
            base_channels=base_channels,
            num_heads=num_heads,
            sigma=sigma,
        )

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> dict:
        return self.encoder.state_dict()

    def get_decoder_state(self) -> dict:
        return self.decoder.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@register_model("angle_delay_equivariant")
def create_angle_delay_equivariant(
    encoded_dim: int = 512,
    base_channels: int = 64,
    num_heads: int = 4,
    sigma: float = 1.0,
    **kwargs,
) -> AngleDelayEquivariant:
    return AngleDelayEquivariant(
        encoded_dim=encoded_dim,
        base_channels=base_channels,
        num_heads=num_heads,
        sigma=sigma,
        **kwargs,
    )
