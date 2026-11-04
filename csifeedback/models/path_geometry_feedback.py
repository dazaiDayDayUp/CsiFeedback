r"""Path-Geometry Feedback：把 CSI 显式表示为少量离格多径的叠加。

物理假设：角-时延域 CSI 主要由若干条 specular path 贡献，每条径在连续角度、
时延上是一个二维复指数；截断到 32×32 网格后，它表现为两个一维 Dirichlet 核的
外积，而不是单个网格点。码字前半部分编码每条径的连续角度、时延与复增益，
后半部分编码漫散射残差。解码器先做可微的离格多径合成，再由卷积残差分支补足
diffuse 分量。

该实现只依赖 ``CSIAutoencoder`` 接口与注册表，不修改任何 baseline 文件。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


__all__ = ["PathGeometryFeedback", "create_path_geometry_feedback"]


_GRID = 32
_TWO_PI = 2.0 * math.pi


class _ConvBN(nn.Sequential):
    """Conv2d + BatchNorm + LeakyReLU 的轻量组合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        negative_slope: float = 0.3,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )


class _PathEncoder(nn.Module):
    """共享卷积主干 + 路径头 + 残差头。"""

    def __init__(
        self,
        num_paths: int,
        residual_dim: int,
        latent_size: int = 4,
        hidden_dim: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.residual_dim = residual_dim
        self.latent_size = latent_size

        c1 = base_channels
        c2 = base_channels + 32
        c3 = base_channels + 64
        c4 = base_channels + 64

        self.stem = _ConvBN(2, c1, 3)
        self.down1 = _ConvBN(c1, c2, 4, stride=2, padding=1)
        self.down2 = _ConvBN(c2, c3, 4, stride=2, padding=1)
        self.down3 = _ConvBN(c3, c4, 4, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.path_head = nn.Sequential(
            nn.Linear(c4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_paths * 4),
        )
        # 小初始化让路径增益从零附近开始，训练更稳定。
        nn.init.normal_(self.path_head[-1].weight, std=0.01)
        nn.init.zeros_(self.path_head[-1].bias)

        if residual_dim > 0:
            pixels = latent_size * latent_size
            if residual_dim % pixels != 0:
                raise ValueError(
                    f"residual_dim={residual_dim} 不能被 latent_size^2={pixels} 整除"
                )
            latent_channels = residual_dim // pixels
            self.residual_head = nn.Conv2d(c4, latent_channels, kernel_size=1)
        else:
            self.residual_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.down1(feat)
        feat = self.down2(feat)
        feat = self.down3(feat)

        raw = self.path_head(self.pool(feat).flatten(1))
        raw = raw.view(-1, self.num_paths, 4)
        angle = torch.sigmoid(raw[..., 0])
        delay = torch.sigmoid(raw[..., 1])
        gain = 0.5 * torch.tanh(raw[..., 2:4])
        path_code = torch.stack((angle, delay, gain[..., 0], gain[..., 1]), dim=-1)
        path_code = path_code.flatten(1)

        if self.residual_head is None:
            return path_code
        residual_code = self.residual_head(feat).flatten(1)
        return torch.cat((path_code, residual_code), dim=1)


class _PathDecoder(nn.Module):
    """离格多径合成 + 残差卷积解码。"""

    def __init__(
        self,
        encoded_dim: int,
        num_paths: int,
        latent_size: int = 4,
        output_activation: str = "tanh",
    ):
        super().__init__()
        self.num_paths = num_paths
        self.path_dim = num_paths * 4
        self.residual_dim = encoded_dim - self.path_dim
        self.latent_size = latent_size
        self.output_activation = output_activation

        if self.residual_dim < 0:
            raise ValueError("num_paths * 4 不能超过 encoded_dim")

        self.register_buffer("_grid_index", torch.arange(_GRID, dtype=torch.float32))
        self.register_buffer("_sum_index", torch.arange(_GRID, dtype=torch.float32))

        if self.residual_dim > 0:
            pixels = latent_size * latent_size
            if self.residual_dim % pixels != 0:
                raise ValueError(
                    f"residual_dim={self.residual_dim} 不能被 latent_size^2={pixels} 整除"
                )
            self.latent_channels = self.residual_dim // pixels
            self.residual_decoder = nn.Sequential(
                nn.ConvTranspose2d(self.latent_channels, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.3, inplace=True),
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.3, inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(32, 2, kernel_size=3, padding=1),
            )
            # 残差分支从零开始，先让物理路径分支主导重建。
            nn.init.zeros_(self.residual_decoder[-1].weight)
            nn.init.zeros_(self.residual_decoder[-1].bias)
        else:
            self.latent_channels = 0
            self.residual_decoder = None

    def _dirichlet(
        self, freq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回一维离格 Dirichlet 核的实部/虚部。

        ``freq`` 形状为 ``(N, K)``，取值在 (0, 1)。返回 ``(N, K, 32)``。
        定义为 ``sum_{k=0}^{31} exp(j 2π (freq - n/32) k) / 32``。
        """
        n = self._grid_index.view(1, 1, 1, _GRID)
        k = self._sum_index.view(1, 1, _GRID, 1)
        phase = _TWO_PI * (freq.unsqueeze(-1).unsqueeze(-1) - n / _GRID) * k
        real = torch.cos(phase).sum(dim=-1) / _GRID
        imag = torch.sin(phase).sum(dim=-1) / _GRID
        return real, imag

    def _synthesize_paths(
        self, path_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = path_params[..., 0]
        delay = path_params[..., 1]
        gain_real = path_params[..., 2]
        gain_imag = path_params[..., 3]

        angle_real, angle_imag = self._dirichlet(angle)
        delay_real, delay_imag = self._dirichlet(delay)

        # 二维核 = delay 核与 angle 核的外积。
        kernel_real = (
            delay_real.unsqueeze(-1) * angle_real.unsqueeze(-2)
            - delay_imag.unsqueeze(-1) * angle_imag.unsqueeze(-2)
        )
        kernel_imag = (
            delay_real.unsqueeze(-1) * angle_imag.unsqueeze(-2)
            + delay_imag.unsqueeze(-1) * angle_real.unsqueeze(-2)
        )

        g_r = gain_real.unsqueeze(-1).unsqueeze(-1)
        g_i = gain_imag.unsqueeze(-1).unsqueeze(-1)
        path_real = (g_r * kernel_real - g_i * kernel_imag).sum(dim=1)
        path_imag = (g_r * kernel_imag + g_i * kernel_real).sum(dim=1)
        return path_real, path_imag

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return 0.5 + 0.5 * torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "hsigmoid":
            return F.relu6(x + 3.0) / 6.0
        raise ValueError(f"未知输出激活：{self.output_activation}")

    def forward(self, code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        path_params = code[:, : self.path_dim].view(-1, self.num_paths, 4)
        path_real, path_imag = self._synthesize_paths(path_params)
        path_out = torch.stack((path_real, path_imag), dim=1)
        path_out = self._activate(path_out)

        if self.residual_decoder is None:
            return path_out, path_out

        residual = code[:, self.path_dim :].view(
            -1, self.latent_channels, self.latent_size, self.latent_size
        )
        residual = self.residual_decoder(residual)
        out = self._activate(
            torch.stack((path_real, path_imag), dim=1) + residual
        )
        return out, path_out


class PathGeometryFeedback(CSIAutoencoder):
    """路径几何反馈自编码器。

    :param encoded_dim: 码字维度（512/256/128/64/32）。
    :param num_paths: 显式编码的多径条数，每条占 4 个实数（角度、时延、复增益）。
    :param latent_size: 残差特征图的空间尺寸；默认 4（4×4）。
    :param hidden_dim: 路径头隐藏维度。
    :param output_activation: ``tanh`` / ``sigmoid`` / ``hsigmoid``。
    """

    def __init__(
        self,
        encoded_dim: int = 512,
        num_paths: int = 8,
        latent_size: int = 4,
        hidden_dim: int = 128,
        base_channels: int = 64,
        output_activation: str = "tanh",
        **kwargs,
    ):
        super().__init__()
        path_dim = num_paths * 4
        if path_dim > encoded_dim:
            raise ValueError(
                f"num_paths={num_paths} 需要 {path_dim} 维，超过 encoded_dim={encoded_dim}"
            )
        self._encoded_dim = encoded_dim
        self.num_paths = num_paths
        self.encoder = _PathEncoder(
            num_paths=num_paths,
            residual_dim=encoded_dim - path_dim,
            latent_size=latent_size,
            hidden_dim=hidden_dim,
            base_channels=base_channels,
        )
        self.decoder = _PathDecoder(
            encoded_dim=encoded_dim,
            num_paths=num_paths,
            latent_size=latent_size,
            output_activation=output_activation,
        )

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> dict:
        return self.encoder.state_dict()

    def get_decoder_state(self) -> dict:
        return self.decoder.state_dict()

    def forward(
        self, x: torch.Tensor, return_path: bool = False
    ):
        code = self.encoder(x)
        out, path_out = self.decoder(code)
        if return_path:
            return out, path_out
        return out


@register_model("path_geometry_feedback")
def create_path_geometry_feedback(
    encoded_dim: int = 512,
    num_paths: int = 8,
    latent_size: int = 4,
    hidden_dim: int = 128,
    base_channels: int = 64,
    output_activation: str = "tanh",
    **kwargs,
) -> PathGeometryFeedback:
    return PathGeometryFeedback(
        encoded_dim=encoded_dim,
        num_paths=num_paths,
        latent_size=latent_size,
        hidden_dim=hidden_dim,
        base_channels=base_channels,
        output_activation=output_activation,
        **kwargs,
    )
