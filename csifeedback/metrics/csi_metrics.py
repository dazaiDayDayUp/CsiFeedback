"""CSI 反馈统一评估指标。"""

import torch
from packaging import version


__all__ = ["compute_nmse", "compute_rho"]


def compute_nmse(
    sparse_pred: torch.Tensor,
    sparse_gt: torch.Tensor,
    zero_point: float = 0.5,
) -> torch.Tensor:
    """在稀疏域（32×32）计算归一化均方误差（NMSE）。

    两个输入张量形状均应为 ``(N, 2, 32, 32)``，其中通道 0 为实部，通道 1 为虚部。
    计算复数功率与误差前，会先用 ``zero_point`` 对数据进行去中心化。
    """
    with torch.no_grad():
        sparse_gt = sparse_gt - zero_point
        sparse_pred = sparse_pred - zero_point

        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10(
            (mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean()
        )
    return nmse


def compute_rho(
    sparse_pred: torch.Tensor,
    raw_gt: torch.Tensor,
    zero_point: float = 0.5,
    nc_expand: int = 257,
    nc_output: int = 125,
) -> torch.Tensor:
    """通过 FFT 扩展计算相关系数 rho。

    ``sparse_pred`` 形状为 ``(N, 2, 32, 32)``；``raw_gt`` 形状为 ``(N, 32, 125, 2)``。
    稀疏预测沿子载波维度零填充到 ``nc_expand`` 个 bin，做 FFT 后切片到 ``nc_output`` 个 bin，
    再计算归一化复数内积。
    """
    with torch.no_grad():
        nt = sparse_pred.size(2)
        nc = sparse_pred.size(3)

        sparse_pred = sparse_pred - zero_point
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # (N, 32, 32, 2)
        n = sparse_pred.size(0)

        zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))

        if version.parse(torch.__version__) > version.parse("1.7.0"):
            sparse_pred = torch.view_as_complex(
                torch.cat((sparse_pred, zeros), dim=2)
            )
            raw_pred = torch.view_as_real(torch.fft.fft(sparse_pred, dim=2))[
                :, :, :nc_output, :
            ]
        else:
            sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
            raw_pred = torch.fft(sparse_pred, signal_ndim=1)[:, :, :nc_output, :]

        norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()
    return rho
