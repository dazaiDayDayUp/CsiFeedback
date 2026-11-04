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
    """共享卷积主干 + 多径峰值热图检测 + 残差头。

    编码器在 32×32 角度-时延网格上输出一张峰值热图和一张复增益图；
    取热图 top-K 峰值，并在每个峰值周围做 3×3 soft-argmax 得到连续角度/时延。
    这样峰值定位由稠密热图负责，比全局回归坐标精确得多。
    """

    def __init__(
        self,
        num_paths: int,
        residual_dim: int,
        latent_size: int = 4,
        hidden_dim: int = 128,
        base_channels: int = 64,
        delay_fft_size: int = 1024,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.residual_dim = residual_dim
        self.latent_size = latent_size
        self.delay_fft_size = delay_fft_size

        c1 = base_channels
        c2 = base_channels + 32
        c3 = base_channels + 64
        c4 = base_channels + 64

        self.stem = _ConvBN(2, c1, 3)
        self.down1 = _ConvBN(c1, c2, 4, stride=2, padding=1)
        self.down2 = _ConvBN(c2, c3, 4, stride=2, padding=1)
        self.down3 = _ConvBN(c3, c4, 4, stride=2, padding=1)

        # 16×16 特征上采样到 32×32，输出峰值热图与复增益图。
        self.path_decoder = nn.Sequential(
            nn.Conv2d(c2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.gain_head = nn.Conv2d(64, 2, kernel_size=1)

        offsets = [(da, dd) for da in (-1, 0, 1) for dd in (-1, 0, 1)]
        self.register_buffer(
            "_offsets", torch.tensor(offsets, dtype=torch.float32)
        )

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

    def _extract_paths(
        self, heatmap: torch.Tensor, gain_map: torch.Tensor
    ) -> torch.Tensor:
        n = heatmap.shape[0]
        probs = torch.softmax(heatmap.flatten(1), dim=1)  # (N, 1024)
        probs2d = probs.view(n, 1, _GRID, _GRID)
        # 3×3 NMS：避免 top-K 在同一个宽峰上重复取点。
        local_max = F.max_pool2d(
            probs2d, kernel_size=3, stride=1, padding=1
        ).eq(probs2d)
        masked = probs2d.masked_fill(~local_max, 0.0)
        _, top_index = torch.topk(
            masked.flatten(1), self.num_paths, dim=1
        )
        angle0 = (top_index // _GRID).float()
        delay0 = (top_index % _GRID).float()

        patches = F.unfold(
            probs.view(n, 1, _GRID, _GRID), kernel_size=3, padding=1
        )  # (N, 9, 1024)
        patch = patches.gather(
            2, top_index.unsqueeze(1).expand(-1, 9, -1)
        )  # (N, 9, K)
        weights = torch.softmax(patch, dim=1)

        offsets = self._offsets  # (9, 2)
        soft_angle = (
            weights * (angle0.unsqueeze(1) + offsets[:, 0].view(1, 9, 1))
        ).sum(dim=1)
        soft_delay = (
            weights * (delay0.unsqueeze(1) + offsets[:, 1].view(1, 9, 1))
        ).sum(dim=1)

        gain_patches = F.unfold(
            gain_map, kernel_size=3, padding=1
        )  # (N, 18, 1024)
        gain_patch = gain_patches.gather(
            2, top_index.unsqueeze(1).expand(-1, 18, -1)
        ).view(n, 2, 9, self.num_paths)
        gains = (gain_patch * weights.unsqueeze(1)).sum(dim=2)  # (N, 2, K)

        path_code = torch.stack(
            (
                soft_angle / _GRID,
                soft_delay / self.delay_fft_size,
                gains[:, 0],
                gains[:, 1],
            ),
            dim=-1,
        )
        return path_code.flatten(1)

    def forward(
        self, x: torch.Tensor, return_maps: bool = False
    ):
        feat16 = self.down1(self.stem(x))
        feat = self.down2(feat16)
        feat = self.down3(feat)

        path_feat = self.path_decoder(feat16)
        heatmap = self.heatmap_head(path_feat)
        gain_map = self.gain_head(path_feat)
        path_code = self._extract_paths(heatmap, gain_map)

        if self.residual_head is None:
            code = path_code
        else:
            residual_code = self.residual_head(feat).flatten(1)
            code = torch.cat((path_code, residual_code), dim=1)

        if return_maps:
            return code, heatmap, gain_map
        return code


class _PathDecoder(nn.Module):
    """离格多径合成 + 残差卷积解码。"""

    def __init__(
        self,
        encoded_dim: int,
        num_paths: int,
        latent_size: int = 4,
        output_activation: str = "identity",
        clamp_output: bool = True,
        delay_fft_size: int = 1024,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.path_dim = num_paths * 4
        self.residual_dim = encoded_dim - self.path_dim
        self.latent_size = latent_size
        self.output_activation = output_activation
        self.clamp_output = clamp_output
        self.delay_fft_size = delay_fft_size

        if self.residual_dim < 0:
            raise ValueError("num_paths * 4 不能超过 encoded_dim")
        if self.delay_fft_size < _GRID:
            raise ValueError("delay_fft_size 不能小于 32")

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
        self, freq: torch.Tensor, grid_size: int, fft_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回一维离格 Dirichlet 核的实部/虚部。

        ``freq`` 形状为 ``(N, K)``，取值在 (0, 1)。返回 ``(N, K, grid_size)``。
        定义为 ``sum_{n=0}^{fft_size-1} exp(j 2π (freq - m/fft_size) n) / fft_size``，
        其中 ``m`` 是保留的网格下标。

        对角度域，``fft_size = 32``（32 根天线全保留）；对时延域，
        ``fft_size = 1024``（只保留前 32 个 delay tap），因此两者不是同一个核。
        用闭式 Dirichlet 公式避免按 1024 点求和。
        """
        grid = torch.arange(
            grid_size, device=freq.device, dtype=freq.dtype
        ).view(1, 1, grid_size)
        delta = freq.unsqueeze(-1) - grid / fft_size

        denom = fft_size * torch.sin(math.pi * delta)
        numer = torch.sin(math.pi * fft_size * delta)
        eps = 1e-6
        small = denom.abs() < eps
        safe_denom = torch.where(small, torch.ones_like(denom), denom)
        magnitude = torch.where(
            small, torch.ones_like(delta), numer / safe_denom
        )

        phase = math.pi * (fft_size - 1) * delta
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return real, imag

    def _synthesize_paths(
        self, path_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = path_params[..., 0]
        delay = path_params[..., 1]
        gain_real = path_params[..., 2]
        gain_imag = path_params[..., 3]

        angle_real, angle_imag = self._dirichlet(angle, _GRID, _GRID)
        delay_real, delay_imag = self._dirichlet(
            delay, _GRID, self.delay_fft_size
        )

        # 数据布局为 (angle, delay)：第一空间维是天线/角度，第二维是子载波/时延。
        kernel_real = (
            angle_real.unsqueeze(-1) * delay_real.unsqueeze(-2)
            - angle_imag.unsqueeze(-1) * delay_imag.unsqueeze(-2)
        )
        kernel_imag = (
            angle_real.unsqueeze(-1) * delay_imag.unsqueeze(-2)
            + angle_imag.unsqueeze(-1) * delay_real.unsqueeze(-2)
        )

        g_r = gain_real.unsqueeze(-1).unsqueeze(-1)
        g_i = gain_imag.unsqueeze(-1).unsqueeze(-1)
        path_real = (g_r * kernel_real - g_i * kernel_imag).sum(dim=1)
        path_imag = (g_r * kernel_imag + g_i * kernel_real).sum(dim=1)
        return path_real, path_imag

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "identity":
            out = x
        elif self.output_activation == "tanh":
            out = 0.5 + 0.5 * torch.tanh(x)
        elif self.output_activation == "sigmoid":
            out = torch.sigmoid(x)
        elif self.output_activation == "hsigmoid":
            out = F.relu6(x + 3.0) / 6.0
        else:
            raise ValueError(f"未知输出激活：{self.output_activation}")
        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def forward(self, code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        path_params = code[:, : self.path_dim].view(-1, self.num_paths, 4)
        path_real, path_imag = self._synthesize_paths(path_params)
        # 路径合成的输出已经是 0.5 去中心后的复系数，直接加回 0.5。
        path_out = torch.stack((path_real + 0.5, path_imag + 0.5), dim=1)
        path_out = self._activate(path_out)

        if self.residual_decoder is None:
            return path_out, path_out

        residual = code[:, self.path_dim :].view(
            -1, self.latent_channels, self.latent_size, self.latent_size
        )
        residual = self.residual_decoder(residual)
        out = self._activate(
            torch.stack((path_real, path_imag), dim=1) + residual + 0.5
        )
        return out, path_out


class PathGeometryFeedback(CSIAutoencoder):
    """路径几何反馈自编码器。

    :param encoded_dim: 码字维度（512/256/128/64/32）。
    :param num_paths: 显式编码的多径条数，每条占 4 个实数（角度、时延、复增益）。
    :param latent_size: 残差特征图的空间尺寸；默认 4（4×4）。
    :param hidden_dim: 路径头隐藏维度。
    :param output_activation: ``identity`` / ``tanh`` / ``sigmoid`` / ``hsigmoid``。
        路径合成已经按复系数尺度输出，默认 ``identity``。
    :param clamp_output: 是否把最终输出裁剪到 [0, 1]；默认开启。
    :param delay_fft_size: 原始时延 DFT 点数；COST2100 为 1024，只保留前 32 个 tap。
    """

    def __init__(
        self,
        encoded_dim: int = 512,
        num_paths: int = 8,
        latent_size: int = 4,
        hidden_dim: int = 128,
        base_channels: int = 64,
        output_activation: str = "identity",
        clamp_output: bool = True,
        delay_fft_size: int = 1024,
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
            delay_fft_size=delay_fft_size,
        )
        self.decoder = _PathDecoder(
            encoded_dim=encoded_dim,
            num_paths=num_paths,
            latent_size=latent_size,
            output_activation=output_activation,
            clamp_output=clamp_output,
            delay_fft_size=delay_fft_size,
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
    output_activation: str = "identity",
    clamp_output: bool = True,
    delay_fft_size: int = 1024,
    **kwargs,
) -> PathGeometryFeedback:
    return PathGeometryFeedback(
        encoded_dim=encoded_dim,
        num_paths=num_paths,
        latent_size=latent_size,
        hidden_dim=hidden_dim,
        base_channels=base_channels,
        output_activation=output_activation,
        clamp_output=clamp_output,
        delay_fft_size=delay_fft_size,
        **kwargs,
    )
