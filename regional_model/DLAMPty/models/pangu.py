from collections.abc import Iterable, Sequence
from math import ceil, floor
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
from torch.profiler import record_function

from utils.constant_mask import LoadConstantMask
# from utils.constant_mask import LoadDummyMask
from utils.types import Shape_T

from .smoothing import SegmentedSmoothingV2

__all__ = ["PanguModel"]


class PanguModel(nn.Module):
    """
    Implementing https://github.com/198808xc/Pangu-Weather
    """

    def __init__(
        self,
        data_spatial_shape: Shape_T,
        upper_vars: int,
        surface_vars: int,
        depths: Sequence[int],
        heads: Sequence[int],
        embed_dim: int,
        patch_shape: Shape_T,
        window_size: Shape_T,
        constant_mask_paths: Optional[list[str]] = None,
        smoothing_kernel_size: Optional[int] = None,
        segmented_smooth: bool = False,
        segmented_smooth_boundary_width: Optional[int] = None,
        residual: bool = False,
        res_conn_after_smooth: bool = True,
    ) -> None:
        """
        Args:
            data_spatial_shape (tuple[int, int, int]): Shape of the input tensor (xZ, xH, xW).
            upper_vars (int): Number of upper-air variables.
            surface_vars (int): Number of surface variables.
            depths (Sequence[int]): Number of blocks in each level, should contain 2 integers.
            heads (Sequence[int]): Number of heads in each level, should contain 2 integers.
            embed_dim (int): Dimension of the patch embedding.
            patch_shape (tuple[int, int, int]): Shape of the patch (pZ, pH, pW).
            window_size (tuple[int, int, int]): window size
            constant_mask_path (Optional[list[str]]): Paths to the constant mask.
            smoothing_kernel_size (Optional[int]): Kernel size for smoothing.
            residual (bool): Whether to add residual connection.
            res_conn_after_smooth (bool): Change the position of residual connection.
        """
        if constant_mask_paths is not None:
            print(f"Using constant masks: {constant_mask_paths}")
            assert len(constant_mask_paths) == 3

        assert len(depths) == 2
        assert len(heads) == 2
        assert embed_dim % heads[0] == 0

        super().__init__()
        # (xZ * xH * xW) pixels are partitioned into (Z * H * W) patches of size (pZ, pH,pW)
        xZ, xH, xW = data_spatial_shape
        pZ, pH, pW = patch_shape
        Z, H, W = ceil(xZ/pZ)+1, ceil(xH/pH), ceil(xW/pW)  # extra Z for the surface data
        drop_path_list = np.linspace(0, 0.2, depths[0]+depths[1])

        if smoothing_kernel_size is not None:
            assert smoothing_kernel_size % 2 == 1
            if segmented_smooth:
                assert segmented_smooth_boundary_width is not None
                smoothing_func = SegmentedSmoothingV2(
                    kernel_size=smoothing_kernel_size,
                    boundary_width=segmented_smooth_boundary_width,
                )
            else:
                smoothing_func = nn.AvgPool3d(
                    kernel_size=(1, smoothing_kernel_size, smoothing_kernel_size),
                    stride=(1, 1, 1),
                    padding=(0, smoothing_kernel_size//2, smoothing_kernel_size//2),
                    count_include_pad=False
                )

            self.smoothing_layer = SmoothingBlock(smoothing_func=smoothing_func)
        else:
            self.smoothing_layer = Identity()

        self.residual = 1 if residual else 0
        self.res_conn_after_smooth = res_conn_after_smooth

        self.patch_embed = PatchEmbedding(
            in_shape=data_spatial_shape,
            dim=embed_dim,
            upper_vars=upper_vars,
            surface_vars=surface_vars,
            patch_shape=patch_shape,
            constant_mask_paths=constant_mask_paths,
        )

        self.layer1 = EarthSpecificLayer(
            in_shape=(Z, H, W),
            dim=embed_dim,
            depth=depths[0],
            heads=heads[0],
            drop_path_ratio=drop_path_list[:depths[0]],  # type: ignore
            window_size=window_size,
        )

        self.layer2 = EarthSpecificLayer(
            in_shape=(Z, ceil(H/2), ceil(W/2)),
            dim=2*embed_dim,
            depth=depths[1],
            heads=heads[1],
            drop_path_ratio=drop_path_list[-depths[1]:],  # type: ignore
            window_size=window_size,
        )

        self.layer3 = EarthSpecificLayer(
            in_shape=(Z, ceil(H/2), ceil(W/2)),
            dim=2*embed_dim,
            depth=depths[1],
            heads=heads[1],
            drop_path_ratio=drop_path_list[-depths[1]:],  # type: ignore
            window_size=window_size,
        )

        self.layer4 = EarthSpecificLayer(
            in_shape=(Z, H, W),
            dim=embed_dim,
            depth=depths[0],
            heads=heads[0],
            drop_path_ratio=drop_path_list[:depths[0]],  # type: ignore
            window_size=window_size,
        )

        self.downsample = DownSample(
            in_shape=(Z, H, W),
            in_channels=embed_dim,
            out_channels=2*embed_dim,
        )

        self.upsample = UpSample(
            in_channels=2*embed_dim,
            out_shape=(Z, H, W),
            out_channels=embed_dim,
        )

        self.patch_recover = PatchRecovery(
            dim=2*embed_dim,
            in_shape=(Z, H, W),
            out_shape=data_spatial_shape,
            upper_vars=upper_vars,
            surface_vars=surface_vars,
            patch_shape=patch_shape,
        )

    def forward(self, input_upper: torch.Tensor, input_surface: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_upper (torch.Tensor): Tensor of shape (B, xZ, xH, xW, C_upper).
            input_surface (torch.Tensor): Tensor of shape (B, xH, xW, C_surface).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of upper-air data and surface data.
        """

        with record_function("Res Save"):
            res_conn_upper = self.residual * input_upper
            res_conn_surface = self.residual * input_surface

        with record_function("Patch Embedding"):
            x = self.patch_embed(input_upper, input_surface)
        with record_function("Transformer Layer 1"):
            x = self.layer1(x)
            skip = x
        with record_function("Down-sample"):
            x = self.downsample(x)
        with record_function("Transformer Layer 2"):
            x = self.layer2(x)
        with record_function("Transformer Layer 3"):
            x = self.layer3(x)
        with record_function("Up-sample"):
            x = self.upsample(x)
        with record_function("Transformer Layer 4"):
            x = self.layer4(x)
            x = torch.cat([skip, x], dim=-1)
        with record_function("Patch Recovering"):
            output_upper, output_surface = self.patch_recover(x)

        if not self.res_conn_after_smooth:
            with record_function("Res Add"):
                output_upper += res_conn_upper
                output_surface += res_conn_surface

        with record_function("Smoothing Layer"):
            output_upper, output_surface = self.smoothing_layer(output_upper, output_surface)

        if self.res_conn_after_smooth:
            with record_function("Res Add"):
                output_upper += res_conn_upper
                output_surface += res_conn_surface

        return output_upper, output_surface


class SmoothingBlock(nn.Module):
    def __init__(
        self,
        smoothing_func: nn.Module,
    ) -> None:
        super().__init__()
        self.smoothing_func = smoothing_func

    def forward(self, x_upper: torch.Tensor, x_surface: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_surface = x_surface.unsqueeze(1)
        x_upper = x_upper.permute(0, 4, 1, 2, 3)  # (B, C_upper, xZ, xH, xW)
        x_surface = x_surface.permute(0, 4, 1, 2, 3)  # (B, C_surface, 1, xH, xW)

        x_upper = self.smoothing_func(x_upper)
        x_surface = self.smoothing_func(x_surface)

        x_upper = x_upper.permute(0, 2, 3, 4, 1)  # (B, xZ, xH, xW, C_upper)
        x_surface = x_surface.permute(0, 2, 3, 4, 1)  # (B, 1, xH, xW, C_surface)
        x_surface = x_surface.squeeze(1)
        return x_upper, x_surface


class Identity(nn.Module):
    """
    Identity layer. Replace smoothing layer when smoothing_kernel_size is None.
    """

    def forward(self, x_upper: torch.Tensor, x_surface: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x_upper, x_surface


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        upper_vars: int,
        surface_vars: int,
        patch_shape: Shape_T,
        constant_mask_paths: Optional[list[str]] = None,
    ) -> None:
        """
        Convert input fields to patches and linearly embed them.

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (xZ, xH, xW).
            dim (int): Dimension of the output embedding.
            upper_vars (int): Number of upper-air variables.
            surface_vars (int): Number of surface variables.
            patch_shape (tuple[int, int, int]): Size of the patch (pZ, pH, pW).
            constant_mask_paths (Optional[list[str]]): Paths to the constant mask.
        """
        super().__init__()
        # Constant masks
        if constant_mask_paths is not None:
            land_mask = LoadConstantMask(constant_mask_paths[0]).unsqueeze(0)  # unsqueeze to add batch dimension
            soil_mask = LoadConstantMask(constant_mask_paths[1]).unsqueeze(0)
            topography_mask = LoadConstantMask(constant_mask_paths[2]).unsqueeze(0)
            # Normalize the topography mask to [0, 1]
            topography_mask = (topography_mask - topography_mask.min()) / \
                (topography_mask.max() - topography_mask.min())
            additional_channels = land_mask.shape[1] + soil_mask.shape[1] + topography_mask.shape[1]
        else:
            land_mask, soil_mask, topography_mask = None, None, None
            additional_channels = 0
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("soil_mask", soil_mask)
        self.register_buffer("topography_mask", topography_mask)

        # Use convolution to partition data into cubes
        self.patch_shape = patch_shape
        self.conv_upper = nn.Conv3d(in_channels=upper_vars,
                                    out_channels=dim,
                                    kernel_size=patch_shape,
                                    stride=patch_shape)

        self.conv_surface = nn.Conv2d(in_channels=surface_vars+additional_channels,
                                      out_channels=dim,
                                      kernel_size=patch_shape[1:],
                                      stride=patch_shape[1:])

        self.upper_pad = GetPad3D(in_shape, patch_shape)
        self.surface_pad = GetPad2D(in_shape[1:], patch_shape[1:])

    def forward(self, input_upper: torch.Tensor, input_surface: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_upper (torch.Tensor): Tensor of shape (B, xZ, xH, xW, C_upper).
            input_surface (torch.Tensor): Tensor of shape (B, xH, xW, C_surface).
        Returns:
            torch.Tensor: Tensor of shape (batch_size, Z*H*W, dim).
        """
        input_upper = input_upper.permute(0, 4, 1, 2, 3)  # (B, C_upper, xZ, xH, xW)
        input_surface = input_surface.permute(0, 3, 1, 2)  # (B, C_sfc, xH, xW)
        if (self.land_mask is not None
                and self.soil_mask is not None
                and self.topography_mask is not None):
            B = input_surface.shape[0]
            input_surface = torch.cat([
                input_surface,
                self.land_mask.repeat(B, 1, 1, 1),
                self.soil_mask.repeat(B, 1, 1, 1),
                self.topography_mask.repeat(B, 1, 1, 1),
            ], dim=1)

        # Pad the input to make it divisible by patch_shape
        input_upper = self.upper_pad(input_upper)
        input_surface = self.surface_pad(input_surface)

        embedding_upper = self.conv_upper(input_upper)  # (B, dim, Z-1, H, W)
        embedding_surface = self.conv_surface(input_surface)  # (B, dim, H, W)

        embedding_surface = embedding_surface.unsqueeze(-3)  # (B, dim, 1, H, W)
        x = torch.cat([embedding_upper, embedding_surface], -3)  # (B, dim, Z, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # (B, Z, H, W, dim)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, Z*H*W, dim)

        return x


class PatchRecovery(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        out_shape: Shape_T,
        upper_vars: int,
        surface_vars: int,
        patch_shape: Shape_T,
    ) -> None:
        """
        Recover the output fields from patches.

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input embedding.
            out_shape (tuple[int, int, int]): (xZ, xH, xW).
            patch_shape (tuple[int, int, int]): Size of the patch (pZ, pH, pW).
        """
        super().__init__()
        self.conv_upper = nn.ConvTranspose3d(
            in_channels=dim,
            out_channels=upper_vars,
            kernel_size=patch_shape,
            stride=patch_shape
        )
        self.conv_surface = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=surface_vars,
            kernel_size=patch_shape[1:],
            stride=patch_shape[1:],
        )
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.upper_crop_idx = GetCrop3DIndex(out_shape, patch_shape)
        self.surface_crop_idx = GetCrop2DIndex(out_shape[1:], patch_shape[1:])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, Z*H*W, dim).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of upper-air data (B, xZ, xH, xW, C_upper) and surface data (B, xH, xW, C_surface
        """
        x = x.permute(0, 2, 1)  # (B, dim, Z*H*W)
        x = x.reshape(x.shape[0], x.shape[1], *self.in_shape)  # (B, dim, Z, H, W)

        x_upper, x_surface = torch.split(x, [x.shape[-3]-1, 1], dim=-3)  # (B, dim, Z-1, H, W), (B, dim, 1, H, W)
        x_surface = torch.squeeze(x_surface, -3)  # (B, dim, H, W)

        # Upsample to original size
        output_upper = self.conv_upper(x_upper)  # (B, C_upper, xZ, xH, xW)
        output_surface = self.conv_surface(x_surface)  # (B, C_surface, xH, xW)

        # Crop to the original size
        output_upper = output_upper[:, :, self.upper_crop_idx[0], self.upper_crop_idx[1], self.upper_crop_idx[2]]
        output_surface = output_surface[:, :, self.surface_crop_idx[0], self.surface_crop_idx[1]]

        output_upper = output_upper.permute(0, 2, 3, 4, 1)  # (B, xZ, xH, xW, C_upper)
        output_surface = output_surface.permute(0, 2, 3, 1)  # (B, xH, xW, C_surface)

        return output_upper, output_surface


class DownSample(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        in_channels: int,
        out_channels: int
    ) -> None:
        """
        Reduces the lateral resolution by a factor of 2.

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.linear = nn.Linear(4*in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(4*in_channels)
        self.Z, self.H, self.W = in_shape
        # Pad to make H and W divisible by 2
        self.pad = GetPad2D((self.H, self.W), (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, Z, H, W, C_in).
        Returns:
            torch.Tensor: Tensor of shape (B, Z, H//2, W//2, C_out).
        """
        x = x.reshape(x.shape[0], self.Z, self.H, self.W, x.shape[-1])  # (B, Z, H, W, C_in)
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)  # (B, Z, H+H%2, W+W%2, C_in)
        x = x.reshape(x.shape[0], self.Z, ceil(self.H/2), 2, ceil(self.W/2),
                      2, x.shape[-1])  # (B, Z, H/2, 2, W/2, 2, C_in)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, Z, H/2, W/2, 2, 2, C_in)
        x = x.reshape(x.shape[0], self.Z * ceil(self.H/2) * ceil(self.W/2), 4*x.shape[-1])  # (B, Z*(H/2)*(W/2), 4*C_in)

        x = self.norm(x)  # (B, Z*(H/2)*(W/2), 4*C_in)
        x = self.linear(x)  # (B, Z*(H/2)*(W/2), C_out)
        return x


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_shape: Shape_T,
        out_channels: int
    ) -> None:
        """
        Increases the lateral resolution by a factor of 2.

        Args:
            in_channels (int): Number of input channels.
            out_shape (tuple[int, int, int]): Shape of the output tensor (Z_out, H_out, W_out).
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, 4*out_channels, bias=False)
        # to mix normalized tensors
        self.linear2 = nn.Linear(out_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.out_shape = out_shape
        self.crop_idx = GetCrop2DIndex(out_shape[1:], (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, Z*H*W, C_in).
        Returns:
            torch.Tensor: Tensor of shape (B, Z_out*H_out*W_out, C_out). (Cropped to output_shape)
        """
        Z_in, H_in, W_in = self.out_shape[0], ceil(self.out_shape[1]/2), ceil(self.out_shape[2]/2)
        x = self.linear(x)  # (B, Z*H*W, 4*C_out)
        x = x.reshape(x.shape[0], Z_in, H_in, W_in, 2, 2, x.shape[-1]//4)  # (B, Z, H, W, 2, 2, C_out)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, Z, H, 2, W, 2, C_out)
        x = x.reshape(x.shape[0], Z_in, H_in * 2, W_in*2, x.shape[-1])  # (B, Z, H*2, W*2, C_out)

        # Crop to output shape
        Z_out, H_out, W_out = self.out_shape
        # assert Z_out == Z_in
        x = x[:, :, self.crop_idx[0], self.crop_idx[1], :]  # (B, Z_out, H_out, W_out, C_out)
        x = x.reshape(x.shape[0], Z_out * H_out * W_out, x.shape[-1])  # (B, Z_out*H_out*W_out, C_out)
        x = self.norm(x)  # (B, Z_out*H_out*W_out, C_out)
        x = self.linear2(x)  # (B, Z_out*H_out*W_out, C_out)
        return x


class EarthSpecificLayer(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        depth: int,
        heads: int,
        drop_path_ratio: Sequence[float],
        window_size: Shape_T,
    ) -> None:
        """
        Basic layer of the network, contains either 2 or 6 blocks

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input token.
            depth (int): Number of blocks in the layer.
            heads (int): Number of heads in the attention layer.
            drop_path_ratio (Sequence[float]): Drop path ratio for each block.
            window_size (tuple[int, int, int]): window size.
        """
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList((EarthSpecificBlock(
            dim=dim,
            in_shape=in_shape,
            drop_path_ratio=drop_path_ratio[i],
            heads=heads,
            roll=(i % 2 == 1),
            window_size=window_size,
        ) for i in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, Z*H*W, dim).
        Returns:
            torch.Tensor: Tensor of shape (batch_size, Z*H*W, dim).
        """
        for i in range(self.depth):
            x = self.blocks[i](x)
        return x


def window_partition(x: torch.Tensor, window_shape: Shape_T) -> torch.Tensor:
    """
    Partition spatial dimensions by window_size and flatten the window dimensions.

    Args:
        x (tensor): (B, Z, H, W, C)
        window_shape (tuple[int, int, int]): attention window's shape (wZ, wH, wW)
    Returns:
        torch.Tensor: Tensor of shape (B * num_windows, wZ, wH, wW, C)
    """
    B, Z, H, W, C = x.shape
    wZ, wH, wW = window_shape
    x = x.reshape(B, Z//wZ, wZ, H//wH, wH, W//wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wZ, wH, wW, C)
    return windows


def window_reverse(windows: torch.Tensor,
                   window_size: Shape_T,
                   Z: int, H: int, W: int) -> torch.Tensor:
    """
    Inverse operation of window_partition.

    Args:
        windows: Tensor of shape (B * num_windows, wZ, wH, wW, C)
        window_size (tuple[int, int, int]): window size
        Z (int): Number of pressure levels.
        H (int): Number of latitude levels.
        W (int): Number of longitude levels.
    Returns:
        torch.Tensor: Tensor of shape (B, Z, H, W, C)
    """
    wZ, wH, wW = window_size
    mZ, mH, mW = ceil(Z/wZ), ceil(H/wH), ceil(W/wW)
    x = windows.reshape(-1, mZ, mH, mW, wZ, wH, wW, windows.shape[-1])
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(-1, wZ*mZ, wH*mH, wW*mW, windows.shape[-1])
    return x


class EarthSpecificBlock(nn.Module):
    def __init__(
        self,
        in_shape: Shape_T,
        dim: int,
        heads: int,
        drop_path_ratio: float,
        window_size: Shape_T,
        roll: bool,
    ) -> None:
        """
        Earth-specific variant of Swin-Transformer, with 3d window attention and earth-specific bias

        Args:
            in_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input token.
            heads (int): Number of heads in the attention layer.
            drop_path_ratio (float): DropPath drop probability.
            window_size (tuple[int, int, int]): window size
            roll (bool): Whether to roll the tensor for half a window size.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i//2 for i in self.window_size)

        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = MLP(dim, 0)
        self.attention = EarthAttention3D(
            dim=dim,
            input_shape=in_shape,
            heads=heads,
            dropout_rate=0,
            window_size=self.window_size,
        )
        self.roll = roll
        self.Z, self.H, self.W = in_shape
        self.pad = GetPad3D(in_shape, window_size)
        self.crop_idx = GetCrop3DIndex(in_shape, window_size)

        # Mask for added padding
        pad_mask = torch.ones((1, self.Z, self.H, self.W, 1))
        pad_mask = self.pad(pad_mask.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        pad_mask = 1 - pad_mask
        if roll:
            pad_mask = pad_mask.roll(shifts=[-self.window_size[i]//2 for i in range(3)], dims=(1, 2, 3))
        pad_mask = window_partition(pad_mask, self.window_size)
        pad_mask = pad_mask.reshape(-1, self.window_size[0]*self.window_size[1]*self.window_size[2])
        pad_mask = pad_mask.unsqueeze(1) + pad_mask.unsqueeze(2)
        pad_mask = pad_mask.masked_fill(pad_mask != 0, float(-100.0))
        pad_mask = pad_mask.masked_fill(pad_mask == 0, float(0.0))

        if roll:
            # Following Swin-Transformer's implementation to calculate attention mask
            img_mask = torch.zeros(1, self.Z, self.H, self.W, 1)
            img_mask = self.pad(img_mask.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            z_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for z in z_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, z, h, w, :] = cnt
                        cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(-1, self.window_size[0]*self.window_size[1]*self.window_size[2])
            nonadjacent_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            nonadjacent_mask = nonadjacent_mask.masked_fill(nonadjacent_mask != 0, float(-100.0))
            nonadjacent_mask = nonadjacent_mask.masked_fill(nonadjacent_mask == 0, float(0.0))

            attention_mask = pad_mask + nonadjacent_mask
            attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0))
        else:
            attention_mask = pad_mask

        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, Z*H*W, dim).
            Z (int): Number of pressure levels.
            H (int): Number of latitude levels.
            W (int): Number of longitude levels.
            roll (bool): Whether to roll the tensor for half a window size.
        Returns:
            torch.Tensor: Tensor of shape (B, Z*H*W, dim).
        """
        shortcut = x
        x = x.reshape(x.shape[0], self.Z, self.H, self.W, x.shape[-1])  # (B, Z, H, W, dim)

        # Pad the input to make it divisible by window_size
        x = x.permute(0, 4, 1, 2, 3)  # (B, dim, Z, H, W)
        x = self.pad(x)  # (B, dim, Z+Z%wZ, H+H%wH, W+W%wW)
        x = x.permute(0, 2, 3, 4, 1)  # (B, Z, H, W, dim)

        og_shape = x.shape
        if self.roll:
            x = x.roll(shifts=[-self.window_size[i]//2 for i in range(3)], dims=(1, 2, 3))

        x = window_partition(x, self.window_size)  # (B * num_windows, wZ, wH, wW, dim)
        x = x.reshape(x.shape[0], int(np.prod(self.window_size)), x.shape[-1])  # (B * num_windows, wZ*wH*wW, dim)

        x = self.attention(x, self.attention_mask)  # (B * num_windows, wZ*wH*wW, dim)

        x = window_reverse(x, self.window_size, self.Z, self.H, self.W)
        # assert x.shape == og_shape

        if self.roll:
            x = x.roll(shifts=[self.window_size[i]//2 for i in range(3)], dims=(1, 2, 3))

        # Crop to original size
        x = x[:, self.crop_idx[0], self.crop_idx[1], self.crop_idx[2], :]  # (B, Z, H, W, dim)

        x = x.reshape(x.shape[0], self.Z*self.H*self.W, x.shape[-1])

        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))
        return x


class EarthAttention3D(nn.Module):
    def __init__(
        self,
        input_shape: Shape_T,
        dim: int,
        heads: int,
        dropout_rate: float,
        window_size: Shape_T
    ) -> None:
        """
        3D window attention layer.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
            dim (int): Dimension of the input token.
            heads (int): Number of heads in the attention layer.
            dropout_rate (float): Dropout rate.
            window_size (tuple[int, int, int]): window size
        """
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        self.heads = heads
        self.dim = dim
        self.scale = (dim//heads) ** -0.5
        self.window_size = window_size

        wZ, wH, wW = window_size
        self.mZ, self.mH, self.mW = map(lambda x, y: ceil(x/y), input_shape, window_size)
        self.window_types = self.mZ * self.mH  # Bias is longitude-independent
        self.earth_specific_bias = torch.zeros((wZ**2 * wH**2 * (2*wW-1), self.window_types, heads))
        # self.earth_specific_bias = nn.Parameter(self.earth_specific_bias, requires_grad=True) #暫時不註解
        # self.earth_specific_bias = nn.init.trunc_normal_(self.earth_specific_bias, std=0.02) #暫時不註解

        self.position_index = self._construct_index()

    def _construct_index(self) -> torch.Tensor:
        """
        Construct the index for reusing symmetrical positional bias.
        """
        wZ, wH, wW = self.window_size

        coords_zi = torch.arange(wZ)
        coords_zj = -torch.arange(wZ)*wZ
        coords_hi = torch.arange(wH)
        coords_hj = -torch.arange(wH)*wH
        coords_w = torch.arange(wW)

        coords_1 = torch.stack(torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij"), dim=0)
        coords_2 = torch.stack(torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij"), dim=0)
        coords_1_flatten = torch.flatten(coords_1, start_dim=1)
        coords_2_flatten = torch.flatten(coords_2, start_dim=1)
        coords = coords_1_flatten.unsqueeze(-1) - coords_2_flatten.unsqueeze(-2)
        coords = coords.permute(1, 2, 0)

        coords[:, :, 2] += wW-1
        coords[:, :, 1] *= 2 * wW - 1
        coords[:, :, 0] *= (2 * wW - 1)*wH*wH

        position_index = coords.sum(-1)
        position_index = position_index.flatten()
        return position_index

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B * wN, wZ*wH*wW, dim).
        Returns:
            torch.Tensor: Tensor of shape (B * wN, wZ*wH*wW, dim).
        """

        original_shape = x.shape
        x = self.qkv(x)  # (B * wN, wZ*wH*wW, dim*3)

        qkv = x.reshape(x.shape[0], x.shape[1], 3,
                        self.heads, self.dim//self.heads).permute(2, 0, 3, 1, 4)  # (3, B * wN, heads, wZ*wH*wW, dim//heads)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        bias = self.earth_specific_bias[self.position_index]
        bias = bias.reshape((self.window_size[0]*self.window_size[1]*self.window_size[2],
                             self.window_size[0]*self.window_size[1]*self.window_size[2],
                             self.window_types,
                             self.heads)).permute(2, 3, 0, 1)

        # Bias is different for each pressure level and latitude, so we need to reshape q@k
        attention = attention.reshape(-1, self.mZ, self.mH, self.mW,
                                      self.heads, attention.shape[-2], attention.shape[-1])
        attention = attention.permute(0, 3, 1, 2, 4, 5, 6)  # (B, mW, mZ, mH, heads, wZ*wH*wW, wZ*wH*wW)
        attention = attention.reshape(-1, self.mZ*self.mH,
                                      self.heads, attention.shape[-2], attention.shape[-1])  # (B*mW, mZ*mH, heads, wZ*wH*wW, wZ*wH*wW)

        bias = bias.to(x.device) # 為什麼？
        attention = attention + bias
        attention = attention.reshape(-1, self.mW, self.mZ, self.mH,
                                      self.heads, attention.shape[-2], attention.shape[-1])
        attention = attention.permute(0, 2, 3, 1, 4, 5, 6)  # (B, mZ, mH, mW, heads, wZ*wH*wW, wZ*wH*wW)
        # (B*mZ*mH*mW, heads, wZ*wH*wW, wZ*wH*wW)
        attention = attention.reshape(-1, self.heads, attention.shape[-2], attention.shape[-1])

        if mask is not None:
            nW = mask.shape[0]
            attention = attention.reshape(-1, nW, self.heads, original_shape[1], original_shape[1])
            attention += mask.unsqueeze(1).unsqueeze(0)
            attention = attention.reshape(-1, self.heads, original_shape[1], original_shape[1])

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        x = (attention @ value).transpose(1, 2).reshape(original_shape)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim*4)
        self.linear2 = nn.Linear(dim*4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


def GetPad3D(img_shape: tuple[int, int, int], sub_shape: tuple[int, int, int]) -> nn.ZeroPad3d:
    """
    Get nn.ZeroPad3d for padding the input to be divisible by sub_shape.

    Args:
        img_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
        sub_shape (tuple[int, int, int]): Shape of the sub tensor (z, h, w).
    Returns:
        nn.ZeroPad3d: Padding layer.
    """
    Z, H, W = img_shape
    z, h, w = sub_shape
    pad = nn.ZeroPad3d((
        floor((-W % w)/2), ceil((-W % w)/2),
        floor((-H % h)/2), ceil((-H % h)/2),
        floor((-Z % z)/2), ceil((-Z % z)/2),
    ))
    return pad


def GetPad2D(img_shape: tuple[int, int], sub_shape: tuple[int, int]) -> nn.ZeroPad2d:
    """
    Get nn.ZeroPad2d for padding the input to be divisible by sub_shape.

    Args:
        img_shape (tuple[int, int]): Shape of the input tensor (H, W).
        sub_shape (tuple[int, int]): Shape of the sub tensor (h, w).
    Returns:
        nn.ZeroPad2d: Padding layer.
    """
    H, W = img_shape
    h, w = sub_shape
    pad = nn.ZeroPad2d((
        floor((-W % w)/2), ceil((-W % w)/2),
        floor((-H % h)/2), ceil((-H % h)/2),
    ))
    return pad


def GetCrop3DIndex(img_shape: tuple[int, int, int], sub_shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    """
    Get the index for reversing padding via GetPad3D.

    Args:
        img_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
        sub_shape (tuple[int, int, int]): Shape of the sub tensor (z, h, w).
    Returns:
        tuple[slice, slice, slice]: Crop index.
    """
    Z, H, W = img_shape
    z, h, w = sub_shape
    return (
        slice(floor((-Z % z)/2), -ceil((-Z % z)/2)) if Z % z != 0 else slice(None),
        slice(floor((-H % h)/2), -ceil((-H % h)/2)) if H % h != 0 else slice(None),
        slice(floor((-W % w)/2), -ceil((-W % w)/2)) if W % w != 0 else slice(None),
    )


def GetCrop2DIndex(img_shape: tuple[int, int], sub_shape: tuple[int, int]) -> tuple[slice, slice]:
    """
    Get the index for reversing padding via GetPad2D.

    Args:
        img_shape (tuple[int, int]): Shape of the input tensor (H, W).
        sub_shape (tuple[int, int]): Shape of the sub tensor (h, w).
    Returns:
        tuple[slice, slice]: Crop index.
    """
    H, W = img_shape
    h, w = sub_shape
    return (
        slice(floor((-H % h)/2), -ceil((-H % h)/2)) if H % h != 0 else slice(None),
        slice(floor((-W % w)/2), -ceil((-W % w)/2)) if W % w != 0 else slice(None),
    )
