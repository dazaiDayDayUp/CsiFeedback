"""CSI 反馈自编码器的抽象基类。"""

from abc import ABC, abstractmethod

import torch.nn as nn


__all__ = ["CSIAutoencoder"]


class CSIAutoencoder(nn.Module, ABC):
    """CLNet、STNet、CsiNet 共享的基类。

    子类必须暴露瓶颈维度，并提供分开保存 encoder/decoder 的辅助方法，
    以满足某些旧版检查点格式的需求。
    """

    @property
    @abstractmethod
    def encoded_dim(self) -> int:
        """瓶颈维度（码字长度）。"""
        raise NotImplementedError

    @abstractmethod
    def get_encoder_state(self) -> dict:
        """返回 encoder 的 ``state_dict``。"""
        raise NotImplementedError

    @abstractmethod
    def get_decoder_state(self) -> dict:
        """返回 decoder 的 ``state_dict``。"""
        raise NotImplementedError

    def load_encoder_state(self, state_dict: dict, strict: bool = True) -> None:
        """将状态加载到 encoder 中。"""
        # 如果 encoder 不是单个 nn.Module，子类可重写此方法。
        encoder = getattr(self, "encoder", None)
        if encoder is None:
            raise NotImplementedError("该模型没有单个 'encoder' 属性")
        encoder.load_state_dict(state_dict, strict=strict)

    def load_decoder_state(self, state_dict: dict, strict: bool = True) -> None:
        """将状态加载到 decoder 中。"""
        decoder = getattr(self, "decoder", None)
        if decoder is None:
            raise NotImplementedError("该模型没有单个 'decoder' 属性")
        decoder.load_state_dict(state_dict, strict=strict)
