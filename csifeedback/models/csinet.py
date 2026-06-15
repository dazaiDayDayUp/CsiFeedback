r"""CsiNet：Wen 等人《Deep Learning for Massive MIMO CSI Feedback》（IEEE WCL 2018）。

PyTorch 复现版本从 ``CsiNet/csinet_pytorch.py`` 迁移而来，层定义与初始化均原样保留。
"""

import torch.nn as nn
import torch.nn.functional as F

from csifeedback.models.base import CSIAutoencoder


__all__ = ["CsiNet", "create_csinet"]


_IMG_HEIGHT = 32
_IMG_WIDTH = 32
_IMG_CHANNELS = 2
_IMG_TOTAL = _IMG_HEIGHT * _IMG_WIDTH * _IMG_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.3)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.3)
        out = self.bn3(self.conv3(out))
        out = F.leaky_relu(out + identity, negative_slope=0.3)
        return out


class CsiNet(CSIAutoencoder):
    def __init__(self, encoded_dim: int = 512):
        super(CsiNet, self).__init__()
        self._encoded_dim = encoded_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(_IMG_TOTAL, encoded_dim)

        self.decoder_fc = nn.Linear(encoded_dim, _IMG_TOTAL)
        self.decoder_res1 = ResidualBlock()
        self.decoder_res2 = ResidualBlock()
        self.decoder_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    def get_encoder_state(self) -> dict:
        return {
            "encoder_conv": self.encoder_conv.state_dict(),
            "encoder_fc": self.encoder_fc.state_dict(),
        }

    def get_decoder_state(self) -> dict:
        return {
            "decoder_fc": self.decoder_fc.state_dict(),
            "decoder_res1": self.decoder_res1.state_dict(),
            "decoder_res2": self.decoder_res2.state_dict(),
            "decoder_conv": self.decoder_conv.state_dict(),
            "sigmoid": self.sigmoid.state_dict(),
        }

    def forward(self, x):
        # x: (N, 2, 32, 32)
        enc = self.encoder_conv(x)
        enc = self.flatten(enc)
        encoded = self.encoder_fc(enc)

        dec = self.decoder_fc(encoded)
        dec = dec.view(-1, _IMG_CHANNELS, _IMG_HEIGHT, _IMG_WIDTH)
        dec = self.decoder_res1(dec)
        dec = self.decoder_res2(dec)
        dec = self.decoder_conv(dec)
        out = self.sigmoid(dec)
        return out


def create_csinet(encoded_dim: int = 512) -> CsiNet:
    """CsiNet 工厂函数。

    :param encoded_dim: 瓶颈维度。常用值：
        512（1/4）、128（1/16）、64（1/32）、32（1/64）。
    """
    return CsiNet(encoded_dim=encoded_dim)
