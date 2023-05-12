from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import torch.nn.functional as F


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


# 定义3D可分离卷积模块
class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=0, dilation=(1,1,1), bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, (1,1,1), (1,1,1), 0, (1,1,1), 1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# 定义新的embed层
class NewEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SeparableConv3d(in_channels, in_channels*2, (3,3,3), (1,1,1), 1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels*2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = SeparableConv3d(in_channels*2, in_channels*4, (3,3,3), (1,1,1), 1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels*4)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = SeparableConv3d(in_channels *4 , out_channels, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        return x

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y

class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class ResDownsamEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.InitConv = InitConv(in_channels=in_channels, out_channels=in_channels*4, dropout=0.2)  # in 4  out 32
        self.Enbolck1 = EnBlock(in_channels=in_channels*4)
        self.conv1 = SeparableConv3d(in_channels * 4, in_channels * 8, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels * 8)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = SeparableConv3d(in_channels * 8, in_channels * 16, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels * 16)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = SeparableConv3d(in_channels * 16, 128, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = SeparableConv3d(128, 32, (3,3,3),(1,1,1),1, bias=False)
        self.BigDown = get_conv_layer(spatial_dims=3, in_channels=4 ,out_channels=out_channels, kernel_size=(4, 4, 4), stride=(4, 4, 4),
                                      dropout=0.0 , conv_only=True, )

    def forward(self, x):
        # (1,4,128,128,128)

        x1_1 = self.InitConv(x)  #(1,16,128,128,128)
        x1_2 = self.BigDown(x)
        x2 = self.Enbolck1(x1_1) #(1,16,128,128,128)

        x3 = self.conv1(x2)  #(1,32,128,128,128)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)

        x4 = self.conv2(x3) #(1,64,128,128,128)
        x4 = self.bn2(x4)
        x4 = self.relu(x4)
        x5 = self.pool(x4) #(1,64,64,64,64)
        # x5_1 =
        x6 = self.conv3(x5) #(1,128,64,64,64)
        x6 = self.bn3(x6)
        x6 = self.pool(x6) #(1,128,32,32,32)


        x7 = self.conv4(x6)
        end = x7+x1_2


        return end


# 定义3D可分离卷积模块
class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=0, dilation=(1,1,1), bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, (1,1,1), (1,1,1), 0, (1,1,1), 1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np


class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        N = kernel_size ** 3
        self.stride = stride
        self.padding = padding
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        self.conv_kernel = nn.Conv3d(in_channels * N, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, N * 3, kernel_size=kernel_size, padding=padding, bias=bias)

        self.mode = "deformable"

    def deformable_mode(self, on=True):  #
        if on:
            self.mode = "deformable"
        else:
            self.mode = "regular"

    def forward(self, x):
        if self.mode == "deformable":
            offset = self.offset_conv_kernel(x)
        else:
            b, c, h, w, d = x.size()
            offset = torch.zeros(b, 3 * self.kernel_size ** 3, h, w, d).to(x)

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]

        # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
        p = p.contiguous().permute(0, 2, 3, 4, 1)  # 5D array

        q_sss = Variable(p.data, requires_grad=False).floor()  # point with all smaller coords
        #         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
        q_lll = q_sss + 1  # all larger coords

        # 8 neighbor points with integer coords
        q_sss = torch.cat([
            torch.clamp(q_sss[..., :N], 0, x.size(2) - 1),  # h_coord
            torch.clamp(q_sss[..., N:2 * N], 0, x.size(3) - 1),  # w_coord
            torch.clamp(q_sss[..., 2 * N:], 0, x.size(4) - 1)  # d_coord
        ], dim=-1).long()
        q_lll = torch.cat([
            torch.clamp(q_lll[..., :N], 0, x.size(2) - 1),  # h_coord
            torch.clamp(q_lll[..., N:2 * N], 0, x.size(3) - 1),  # w_coord
            torch.clamp(q_lll[..., 2 * N:], 0, x.size(4) - 1)  # d_coord
        ], dim=-1).long()
        q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)

        # (b, h, w, d, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:2 * N].lt(self.padding) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding),
            p[..., 2 * N:].lt(self.padding) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))  # все еще непонятно, что тут происходит за wtf
        p = p * (1 - mask) + floor_p * mask

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)

        # trilinear kernel (b, h, w, d, N)
        g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        # get values in all 8 neighbor points
        # (b, c, h, w, d, N) - 6D-array
        x_q_sss = self._get_x_q(x, q_sss, N)
        x_q_lll = self._get_x_q(x, q_lll, N)
        x_q_ssl = self._get_x_q(x, q_ssl, N)
        x_q_sls = self._get_x_q(x, q_sls, N)
        x_q_sll = self._get_x_q(x, q_sll, N)
        x_q_lss = self._get_x_q(x, q_lss, N)
        x_q_lsl = self._get_x_q(x, q_lsl, N)
        x_q_lls = self._get_x_q(x, q_lls, N)

        # (b, c, h, w, d, N)
        x_offset = g_sss.unsqueeze(dim=1) * x_q_sss + \
                   g_lll.unsqueeze(dim=1) * x_q_lll + \
                   g_ssl.unsqueeze(dim=1) * x_q_ssl + \
                   g_sls.unsqueeze(dim=1) * x_q_sls + \
                   g_sll.unsqueeze(dim=1) * x_q_sll + \
                   g_lss.unsqueeze(dim=1) * x_q_lss + \
                   g_lsl.unsqueeze(dim=1) * x_q_lsl + \
                   g_lls.unsqueeze(dim=1) * x_q_lls

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing='ij')

        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset

        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()

        #           (0, 1, 2, 3, 4)
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)

        return x_offset


def deform_conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return DeformConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DeformBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DeformBasicBlock, self).__init__()
        self.conv1 = deform_conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = deform_conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape

    def forward(self, input):
        return input.view((-1,) + self.shape)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out





# 定义新的embed层
class NewEmbedding1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SeparableConv3d(in_channels, in_channels*2, (3,3,3), (1,1,1), 1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels*2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = SeparableConv3d(in_channels*2, in_channels*4, (3,3,3), (1,1,1), 1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels*4)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = DeformConv3d(in_channels=in_channels*4, out_channels=out_channels, kernel_size=2, stride=1, padding=1, bias=False) # in_channels=16, out_channels=32
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)

        return x

# 导入SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1, 1)
        return x * y


class NewEmbedding4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SeparableConv3d(in_channels, in_channels*2, (3,3,3), (1,1,1), 1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels*2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = SeparableConv3d(in_channels*2, in_channels*4, (3,3,3), (1,1,1), 1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels*4)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = SeparableConv3d(in_channels *4 , out_channels, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        # 添加SE模块
        self.se1 = SELayer(in_channels*2)
        self.se2 = SELayer(in_channels*4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 添加SE模块
        x = self.se1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 添加SE模块
        x = self.se2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        return x