r"""Angular-Spectral Feedback：角域谱混合 + 时延局部卷积的 CSI 自编码器。

物理先验：
- 角域来自 32 点 DFT，是周期域；多径在角域是若干平面波，适合用 FFT 低阶
  谱模式做全局混合。
- 时延域是 1024 点 DFT 截断到前 32 个 tap，是截断边界；用带零 padding 的
  一维卷积在时延方向做局部/膨胀混合，避免把时延维当成周期。

因此本模型在瓶颈特征上使用“角度维 FFT 低阶谱卷积 + 时延维局部卷积”的
谱-空间块，既保留角域全局平面波叠加，又尊重时延截断。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from csifeedback.models.base import CSIAutoencoder
from csifeedback.models.registry import register_model


__all__ = ["AngularSpectralFeedback", "create_angular_spectral_feedback"]


class AngularSpectralConv(nn.Module):
    """角度维低阶谱卷积 + 时延维局部卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        angle_size: int,
        modes: int = 5,
    ):
        super().__init__()
        self.angle_size = angle_size
        self.modes = modes
        self.weight = nn.Parameter(
            torch.randn(
                in_channels, out_channels, modes, dtype=torch.cfloat
            )
            * 0.02
        )
        self.delay_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=False,
        )
        self.delay_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, height, width = x.shape
        x_ft = torch.fft.rfft(x, dim=2)
        freq = x_ft.shape[2]
        m = min(self.modes, freq)
        out_ft = torch.zeros(
            n,
            self.weight.shape[1],
            freq,
            width,
            dtype=x_ft.dtype,
            device=x.device,
        )
        out_ft[:, :, :m, :] = torch.einsum(
            "bihw,ioh->bohw",
            x_ft[:, :, :m, :],
            self.weight[:, :, :m],
        )
        out = torch.fft.irfft(out_ft, n=height, dim=2)
        out = self.delay_conv(out)
        return self.delay_norm(out)


class SpectralBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        angle_size: int,
        modes: int = 5,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.spectral = AngularSpectralConv(
            channels, channels, angle_size=angle_size, modes=modes
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spectral(self.norm(x))
        h = self.pointwise(self.act(h))
        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        base_channels: int = 64,
        modes: int = 5,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels + 32
        c3 = base_channels + 64
        self.stem = nn.Sequential(
            nn.Conv2d(2, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
        )
        self.spectral = SpectralBlock(c3, angle_size=8, modes=modes)
        self.reduce = nn.Conv2d(c3, 16, 1)
        self.fc = nn.Linear(16 * 8 * 8, encoded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.spectral(x)
        x = self.reduce(x)
        return self.fc(x.flatten(1))


class Decoder(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        base_channels: int = 64,
        modes: int = 5,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels + 32
        self.fc = nn.Linear(encoded_dim, 16 * 8 * 8)
        self.expand = nn.Sequential(
            nn.Conv2d(16, c1, 1),
            nn.GroupNorm(8, c1),
            nn.GELU(),
        )
        self.spectral1 = SpectralBlock(c1, angle_size=8, modes=modes)
        self.spectral2 = SpectralBlock(c1, angle_size=8, modes=modes)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.final = nn.Conv2d(c1, 2, 3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        x = self.fc(code).view(-1, 16, 8, 8)
        x = self.expand(x)
        x = self.spectral1(x)
        x = self.spectral2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return self.act(x)


class AngularSpectralFeedback(CSIAutoencoder):
    """角度谱混合 + 时延局部卷积的 CSI 反馈自编码器。"""

    def __init__(
        self,
        encoded_dim: int = 512,
        base_channels: int = 64,
        modes: int = 5,
        **kwargs,
    ):
        super().__init__()
        self._encoded_dim = encoded_dim
        self.encoder = Encoder(
            encoded_dim=encoded_dim,
            base_channels=base_channels,
            modes=modes,
        )
        self.decoder = Decoder(
            encoded_dim=encoded_dim,
            base_channels=base_channels,
            modes=modes,
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


@register_model("angular_spectral_feedback")
def create_angular_spectral_feedback(
    encoded_dim: int = 512,
    base_channels: int = 64,
    modes: int = 5,
    **kwargs,
) -> AngularSpectralFeedback:
    return AngularSpectralFeedback(
        encoded_dim=encoded_dim,
        base_channels=base_channels,
        modes=modes,
        **kwargs,
    )
