# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Type

import torch.nn.functional

from .adapter_modules import ExtractorCMAttn, InjectorCMAttn, InteractionBlock, SpatialPriorModule, deform_inputs, InjectorAdd, ExtractorAdd, InjectorCMX, ExtractorCMX

from .common import LayerNorm2d, MLPBlock
from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
import torchvision

class MLPBlockModEnc(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        interm_embeddings=[]
        for blk in self.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        
        return x, interm_embeddings


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
        


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class InteractionBlockSam(InteractionBlock):
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        bs, _,dim = x.shape
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                            feat=c, spatial_shapes=deform_inputs1[1],
                            level_start_index=deform_inputs1[2])
        x = x.permute(0,2,1).view(bs,dim,H,W).permute(0,2,3,1)
        with torch.no_grad():
            for idx, blk in enumerate(blocks):
                x = blk(x)
        x = x.permute(0,3,1,2).view(bs,dim,-1).permute(0,2,1)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                            feat=x, spatial_shapes=deform_inputs2[1],
                            level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                                feat=x, spatial_shapes=deform_inputs2[1],
                                level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c
    
class InteractionBlockSamCMAttn(InteractionBlock):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=..., drop=0, drop_path=0, with_cffn=True, cffn_ratio=0.25, init_values=0, deform_ratio=1, extra_extractor=False, with_cp=False):
        super().__init__(dim, num_heads, n_points, norm_layer, drop, drop_path, with_cffn, cffn_ratio, init_values, deform_ratio, extra_extractor, with_cp)
        self.injector = InjectorCMAttn(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = ExtractorCMAttn(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                ExtractorCMAttn(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])

    def forward(self, x, c, blocks, H, W):
        bs, _,dim = x.shape
        x = self.injector(query=x,
                            feat=c)
        x = x.permute(0,2,1).view(bs,dim,H,W).permute(0,2,3,1)
        with torch.no_grad():
            for idx, blk in enumerate(blocks):
                x = blk(x)
        x = x.permute(0,3,1,2).view(bs,dim,-1).permute(0,2,1)
        c = self.extractor(query=c, 
                            feat=x, 
                             H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c,
                                feat=x, 
                                 H=H, W=W)
        return x, c
    


class InteractionBlockSamCMX(InteractionBlock):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=..., drop=0, drop_path=0, with_cffn=True, cffn_ratio=0.25, init_values=0, deform_ratio=1, extra_extractor=False, with_cp=False):
        super().__init__(dim, num_heads, n_points, norm_layer, drop, drop_path, with_cffn, cffn_ratio, init_values, deform_ratio, extra_extractor, with_cp)
        self.injector = InjectorCMX(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = ExtractorCMX(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                ExtractorCMX(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])

    def forward(self, x, c, blocks, H, W):
        bs, _,dim = x.shape
        x = self.injector(query=x,
                            feat=c, H=H, W=W)
        x = x.permute(0,2,1).view(bs,dim,H,W).permute(0,2,3,1)
        with torch.no_grad():
            for idx, blk in enumerate(blocks):
                x = blk(x)
        x = x.permute(0,3,1,2).view(bs,dim,-1).permute(0,2,1)
        c = self.extractor(query=c, 
                            feat=x, 
                             H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c,
                                feat=x, 
                                 H=H, W=W)
        return x, c

    
class InteractionBlockSamAdd(InteractionBlock):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=..., drop=0, drop_path=0, with_cffn=True, cffn_ratio=0.25, init_values=0, deform_ratio=1, extra_extractor=False, with_cp=False):
        super().__init__(dim, num_heads, n_points, norm_layer, drop, drop_path, with_cffn, cffn_ratio, init_values, deform_ratio, extra_extractor, with_cp)
        self.injector = InjectorAdd(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = ExtractorAdd(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                ExtractorAdd(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        bs, _,dim = x.shape
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                            feat=c, spatial_shapes=deform_inputs1[1],
                            level_start_index=deform_inputs1[2])
        x = x.permute(0,2,1).view(bs,dim,H,W).permute(0,2,3,1)
        with torch.no_grad():
            for idx, blk in enumerate(blocks):
                x = blk(x)
        x = x.permute(0,3,1,2).view(bs,dim,-1).permute(0,2,1)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                            feat=x, spatial_shapes=deform_inputs2[1],
                            level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                                feat=x, spatial_shapes=deform_inputs2[1],
                                level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c

class ImageEncoderViTAdapter(ImageEncoderViT):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, out_chans: int = 256, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, global_attn_indexes: Tuple[int, ...] = ()) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes)
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        use_abs_pos_emb=False
        use_rel_pos_bias=True
        init_values=1e-6
        drop_path_rate=0.3
        conv_inplane=64
        n_points=4
        deform_num_heads=16
        cffn_ratio=0.25
        deform_ratio=0.5
        with_cp=False
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        if depth == 24:
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        elif depth == 12:
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
        elif depth == 32:
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]]
        use_extra_extractor=True
        with_cffn=True
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = True
        self.fusion = False

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockSam(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0.3,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x: torch.Tensor, depth) -> torch.Tensor:
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        c1, c2, c3, c4 = self.spm(depth)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x = self.patch_embed(x)
        bs, H, W, dim = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = x.permute(0,3,1,2).flatten(2).transpose(1, 2)
        # outs=[]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            
            # outs.append(x)
        
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        x = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
        # if self.add_vit_feature:
        #     x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
        #     x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        #     c4 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        #     c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x3

        # Final Norm
        # f1 = self.norm1(c1)
        # f2 = self.norm2(c2)
        # f3 = self.norm3(c3)
        # f4 = self.norm4(c4)

        # x = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.neck(x)
        
        return x, None #[f1, f2, f3, f4]

    
class ImageEncoderViTAdd(ImageEncoderViT):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, out_chans: int = 256, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, global_attn_indexes: Tuple[int, ...] = ()) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes)
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        use_abs_pos_emb=False
        use_rel_pos_bias=True
        init_values=1e-6
        drop_path_rate=0.3
        conv_inplane=64
        n_points=4
        deform_num_heads=16
        cffn_ratio=0.25
        deform_ratio=0.5
        with_cp=False
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        if depth == 24:
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        elif depth == 12:
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
        elif depth == 32:
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]]
        use_extra_extractor=True
        with_cffn=True
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = True
        self.fusion = False

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockSamAdd(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0.3,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x: torch.Tensor, depth) -> torch.Tensor:
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        c1, c2, c3, c4 = self.spm(depth)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        # c = torch.cat([c2, c3, c4], dim=1)

        x = self.patch_embed(x)
        bs, H, W, dim = x.shape
        c = F.interpolate(c1, (H,W), mode='bilinear')
        c = c.flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = x.permute(0,3,1,2).flatten(2).transpose(1, 2)
        # outs=[]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            
            # outs.append(x)
        
        # Split & Reshape
        # c2 = c[:, 0:c2.size(1), :]
        # c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        # c4 = c[:, c2.size(1) + c3.size(1):, :]

        # c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        # c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        # c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        # c1 = self.up(c2) + c1

        x = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
        # if self.add_vit_feature:
        #     x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
        #     x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        #     c4 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        #     c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x3

        # Final Norm
        # f1 = self.norm1(c1)
        # f2 = self.norm2(c2)
        # f3 = self.norm3(c3)
        # f4 = self.norm4(c4)

        # x = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.neck(x)
        
        return x, None #[f1, f2, f3, f4]
    



class MLPBlockResidual(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        residual: bool = False
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()
        self.residual = residual
        if self.residual:
            self.sigma = nn.Parameter(0 * torch.ones((embedding_dim)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.residual:
            return self.lin2(self.act(self.lin1(x)))
        else:
            return x + self.sigma * self.lin2(self.act(self.lin1(x)))
    

class ImageEncoderViTAdapterRGB(ImageEncoderViT):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, out_chans: int = 256, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, global_attn_indexes: Tuple[int, ...] = ()) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes)
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        if depth == 24:
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        elif depth == 12:
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
        elif depth == 32:
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]]
        
        
        self.adapter = []
        for i in range(depth):
            self.adapter.append(MLPBlockResidual(embed_dim, embed_dim//4, residual=True))
        self.adapter = nn.ModuleList(self.adapter)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        interm_embeddings=[]
        for blk, adapter_layer in zip(self.blocks,self.adapter):
            x = blk(x)
            x = adapter_layer(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        
        return x, interm_embeddings


class ImageEncoderViTAdapterResEnc(ImageEncoderViT):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, out_chans: int = 256, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, global_attn_indexes: Tuple[int, ...] = ()) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes)
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        use_abs_pos_emb=False
        use_rel_pos_bias=True
        init_values=1e-6
        drop_path_rate=0.3
        conv_inplane=64
        n_points=4
        deform_num_heads=16
        cffn_ratio=0.25
        deform_ratio=0.5
        with_cp=False
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        if depth == 24:
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        elif depth == 12:
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
        elif depth == 32:
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]]
        use_extra_extractor=True
        with_cffn=True
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = True
        self.fusion = False

        # self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        resnet_enc = torchvision.models.resnet18(pretrained=False, progress=False)
        resnet_enc.load_state_dict(torch.load('pretrained_checkpoint/resnet18-f37072fd.pth'))
        self.mlp_layer_mod_enc = MLPBlockModEnc(512,embed_dim,256)
        self.modality_encoder = nn.Sequential(
            resnet_enc.conv1,
            resnet_enc.bn1,
            resnet_enc.relu,
            resnet_enc.maxpool,
            resnet_enc.layer1,
            resnet_enc.layer2,
            resnet_enc.layer3,
            resnet_enc.layer4
        )
        self.interactions = nn.Sequential(*[
            InteractionBlockSamCMAttn(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0.3,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.interactions.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, depth) -> torch.Tensor:

        x = self.patch_embed(x)
        bs, H, W, dim = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = x.permute(0,3,1,2).flatten(2).transpose(1, 2)


        c = self.modality_encoder(depth)
        c = torch.nn.functional.interpolate(c, (H,W), mode='bilinear')
        c = c.flatten(2).transpose(1,2)
        c = self.mlp_layer_mod_enc(c)

        # outs=[]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], H, W)
            

        x = x.transpose(1, 2).view(bs, dim, H, W).contiguous()

        x = self.neck(x)
        
        return x, None 
    

class ImageEncoderViTAdapterViTEnc(ImageEncoderViT):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, out_chans: int = 256, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, global_attn_indexes: Tuple[int, ...] = (), me_layers: int = 4) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes)
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        use_abs_pos_emb=False
        use_rel_pos_bias=True
        init_values=1e-6
        drop_path_rate=0.3
        conv_inplane=64
        n_points=4
        deform_num_heads=16
        cffn_ratio=0.25
        deform_ratio=0.5
        with_cp=False
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        if depth == 24:
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]  # 6
        elif depth == 12:
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]  # 4
        elif depth == 32:
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]]  # 8
        use_extra_extractor=True
        with_cffn=True
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = True
        self.fusion = False


        self.patch_embed_modality = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.modality_encoder = nn.ModuleList()
        modality_encoder_depth = me_layers
        for i in range(modality_encoder_depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i != modality_encoder_depth-1 else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.modality_encoder.append(block)

        self.interactions = nn.Sequential(*[
            InteractionBlockSamCMAttn(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0.3,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.interactions.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, depth) -> torch.Tensor:

        x = self.patch_embed(x)
        bs, H, W, dim = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = x.permute(0,3,1,2).flatten(2).transpose(1, 2)

        c = self.patch_embed_modality(depth)
        if self.pos_embed is not None:
            c = c + self.pos_embed
        for layer in self.modality_encoder:
            c = layer(c)
        c = c.permute(0,3,1,2).flatten(2).transpose(1, 2)

        # outs=[]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], H, W)
            

        x = x.transpose(1, 2).view(bs, dim, H, W).contiguous()

        x = self.neck(x)
        
        return x, x 


class ImageEncoderViTAdapterViTEncCMX(ImageEncoderViT):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4, out_chans: int = 256, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, global_attn_indexes: Tuple[int, ...] = (), me_layers: int = 4) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans, qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size, global_attn_indexes)
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        use_abs_pos_emb=False
        use_rel_pos_bias=True
        init_values=1e-6
        drop_path_rate=0.3
        conv_inplane=64
        n_points=4
        deform_num_heads=16
        cffn_ratio=0.25
        deform_ratio=0.5
        with_cp=False
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]
        if depth == 24:
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]  # 6
        elif depth == 12:
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]  # 4
        elif depth == 32:
            interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]]  # 8
        use_extra_extractor=True
        with_cffn=True
        self.interaction_indexes = interaction_indexes
        self.fusion = False


        self.patch_embed_modality = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.modality_encoder = nn.ModuleList()
        modality_encoder_depth = me_layers
        for i in range(modality_encoder_depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i != modality_encoder_depth-1 else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.modality_encoder.append(block)

        self.interactions = nn.Sequential(*[
            InteractionBlockSamCMX(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0.3,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.interactions.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, depth) -> torch.Tensor:

        x = self.patch_embed(x)
        bs, H, W, dim = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = x.permute(0,3,1,2).flatten(2).transpose(1, 2)

        c = self.patch_embed_modality(depth)
        if self.pos_embed is not None:
            c = c + self.pos_embed
        for layer in self.modality_encoder:
            c = layer(c)
        c = c.permute(0,3,1,2).flatten(2).transpose(1, 2)

        # outs=[]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], H, W)
            

        x = x.transpose(1, 2).view(bs, dim, H, W).contiguous()

        x = self.neck(x)
        
        return x, x 