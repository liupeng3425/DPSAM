# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, ImageEncoderViTAdapter, ImageEncoderViTAdd, ImageEncoderViTAdapterRGB, ImageEncoderViTAdapterResEnc, ImageEncoderViTAdapterViTEnc, ImageEncoderViTAdapterViTEncCMX

# def get_modality_enc_layers(model_type: str):
#     layer_num = model_type.split('layer_')[-1].split('_')[0]
#     return int(layer_num)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

def build_sam_vit_h_adapter(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        depth_data=True,
    )

def build_sam_vit_h_add(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        depth_data=True,
        use_add_fusion=True,
    )


def build_sam_vit_h_adapt_rgb(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        backbone=ImageEncoderViTAdapterRGB
    )

def build_sam_vit_h_adapter_rn(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        backbone=ImageEncoderViTAdapterResEnc
    )

def build_sam_vit_h_adapter_vit(checkpoint=None, me_layers=4):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        backbone=partial(ImageEncoderViTAdapterViTEnc, me_layers=me_layers)
    )

build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

def build_sam_vit_l_adapter(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        depth_data=True
    )

def build_sam_vit_l_add(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        depth_data=True,
        use_add_fusion=True,
    )

def build_sam_vit_l_adapt_rgb(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        backbone=ImageEncoderViTAdapterRGB
    )


def build_sam_vit_l_adapter_rn(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        backbone=ImageEncoderViTAdapterResEnc
    )

def build_sam_vit_l_adapter_vit(checkpoint=None, me_layers=4):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        backbone=partial(ImageEncoderViTAdapterViTEnc, me_layers=me_layers)
    )


def build_sam_vit_l_adapter_vit_cmx(checkpoint=None, me_layers=4):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        backbone=partial(ImageEncoderViTAdapterViTEncCMX, me_layers=4)
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_vit_b_adapter(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        depth_data=True,
    )

def build_sam_vit_b_add(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        depth_data=True,
        use_add_fusion=True,
    )

def build_sam_vit_b_adapt_rgb(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        backbone=ImageEncoderViTAdapterRGB
    )

def build_sam_vit_b_adapter_rn(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        backbone=ImageEncoderViTAdapterResEnc
    )

def build_sam_vit_b_adapter_vit(checkpoint=None, me_layers=4):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        backbone=partial(ImageEncoderViTAdapterViTEnc, me_layers=me_layers)
    )


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_h_adapter": build_sam_vit_h_adapter,
    "vit_h_add": build_sam_vit_h_add,
    "vit_h_adapt_rgb": build_sam_vit_h_adapt_rgb,
    "vit_h_adapter_rn": build_sam_vit_h_adapter_rn,
    "vit_h_adapter_vit": build_sam_vit_h_adapter_vit,
    "vit_l": build_sam_vit_l,
    "vit_l_adapter": build_sam_vit_l_adapter,
    "vit_l_add": build_sam_vit_l_add,
    "vit_l_adapt_rgb": build_sam_vit_l_adapt_rgb,
    "vit_l_adapter_rn": build_sam_vit_l_adapter_rn,
    "vit_l_adapter_vit": partial(build_sam_vit_l_adapter_vit, me_layers=4),
    "vit_l_adapter_vit_cmx": build_sam_vit_l_adapter_vit_cmx,
    "vit_b": build_sam_vit_b,
    "vit_b_adapter": build_sam_vit_b_adapter,
    "vit_b_add": build_sam_vit_b_add,
    "vit_b_adapt_rgb": build_sam_vit_b_adapt_rgb,
    "vit_b_adapter_rn": build_sam_vit_b_adapter_rn,
    "vit_b_adapter_vit": build_sam_vit_b_adapter_vit,
}
for layer_num in [1,2,3,4]:
    sam_model_registry[f'vit_l_adapter_vit_layer_{layer_num}'] = partial(build_sam_vit_l_adapter_vit, me_layers=layer_num)
    sam_model_registry[f'vit_h_adapter_vit_layer_{layer_num}'] = partial(build_sam_vit_h_adapter_vit, me_layers=layer_num)
    sam_model_registry[f'vit_b_adapter_vit_layer_{layer_num}'] = partial(build_sam_vit_b_adapter_vit, me_layers=layer_num)


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    depth_data=False,
    use_add_fusion=False, 
    backbone=None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    if backbone is not None:
        img_encoder = backbone
    else:
        # 兼容以前的代码
        if depth_data:
            if use_add_fusion:
                img_encoder = ImageEncoderViTAdd
            else:
                img_encoder = ImageEncoderViTAdapter
        else:
            img_encoder = ImageEncoderViT
    
    sam = Sam(
        image_encoder=img_encoder(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=False)
    return sam
