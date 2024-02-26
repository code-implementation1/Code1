# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# This file refers to https://github.com/open-mmlab/mmcv/blob/v1.5.0/

import math
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F

from decode_head import BaseDecodeHead
from utils_mmcv import build_norm_layer, build_activation_layer, nlc_to_nchw, nchw_to_nlc, \
            ConvModule, DepthwiseSeparableConvModule, TokenEmbedding_noCoord_seg, \
            TokenUpdator, AdaptiveFeatureSelector_1_layers

class SegDecodingTransformer(BaseDecodeHead):
    """SegDecodingTransformer
    """
    def __init__(self, num_heads=4, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True, mlp_ratio=4,
                 ln_norm_cfg=None, all_levels=True, ghost_up=False, mask_loss=False,
                 ratio=math.sqrt(2), div_loss=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.all_levels = all_levels
        self.ghost_up = ghost_up
        self.mask_loss = mask_loss
        self.div_loss = div_loss

        self.laterals = nn.ModuleList()
        for in_channel in self.in_channels:
            self.laterals.append(nn.Conv2d(in_channel, self.channels, 1, bias=False))

        self.transfer = nn.ModuleList()
        for _ in self.in_channels[:-1]:
            self.transfer.append(
                FeaTransLayer(self.channels, num_heads, self.num_classes, attn_drop_rate,
                              drop_rate, qkv_bias, mlp_ratio, ln_norm_cfg=ln_norm_cfg))

        if all_levels:
            self.proj = ConvModule(len(self.in_channels) * self.channels, self.channels,
                                       3, 1, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                                       act_cfg=self.act_cfg)
        if ghost_up:
            self.ghost_up_layers = nn.ModuleList()  # [2, 4, 8]
            for i, _ in enumerate(self.in_channels[:-1]):
                self.ghost_up_layers.append(
                    GhostUpscale(self.channels, 2 ** (i + 1),
                        norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        self.token_emb = nn.ModuleList()
        self.gated_tokens = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.token_emb.append(
                TokenEmbedding_noCoord_seg(num_tokens=self.num_classes,
                    out_dim=self.channels, ratio=ratio))
        for i in range(len(self.in_channels)-1):
            self.gated_tokens.append(
                TokenUpdator(in_channels=self.channels, feat_channels=self.channels,
                            out_channels=self.channels,
                            act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')))

        self.learner = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, groups=num_heads, bias=False),  # 256 -> 256
            nn.Conv2d(self.channels, self.num_classes, 1, bias=False)  # 256 -> 150
        )

        self.trans = nn.Linear(self.channels, self.num_classes)

        self.layer_slcts = nn.ModuleList()
        self.norms = nn.ModuleList()
        for in_channel in self.in_channels:
            self.layer_slcts.append(
                AdaptiveFeatureSelector_1_layers(in_channel,
                    norm_cfg=dict(type='SyncBN', requires_grad=True)))
            self.norms.append(build_norm_layer(dict(type='LN', requires_grad=True), in_channel)[1])

    def forward(self, inputs):
        inters, new_inputs = [], []
        for i in range(len(inputs)):
            layers_output, out = self.layer_slcts[i](inputs[i])  # nchw
            hw_shape = inputs[i].shape[2:]
            out = self.norms[i](nchw_to_nlc(out))
            new_inputs.append(nlc_to_nchw(out, hw_shape))
            inters.append(layers_output)
        inputs = new_inputs
        del new_inputs
        inputs = self._transform_inputs(inputs)
        laterals = [layer(inputs[i]) for i, layer in enumerate(self.laterals)]

        last_mask = self.learner(laterals[-1])
        kc = self.token_emb[-1](last_mask)  # bkc

        masks = [last_mask]
        outs = [laterals[-1]]
        for i in range(len(self.transfer), 0, -1):
            low = laterals[i - 1]
            _, _, h, w = low.shape
            trans_outs = self.transfer[i - 1](low, kc)
            res, attn = trans_outs['out'], trans_outs['attn']  # bchw, blk

            mask = nlc_to_nchw(attn, (h, w))  # bkhw
            tokens = self.token_emb[i - 1](mask)  # bkc
            kc = self.gated_tokens[i - 1](tokens, kc)
            masks.append(mask)
            outs.append(res)

        if self.all_levels:
            if self.ghost_up:
                outs.reverse()
                for i, up_layer in enumerate(self.ghost_up_layers):
                    outs[i + 1] = up_layer(outs[i + 1])
            h, w = inputs[0].shape[2:]
            outs = [F.interpolate(out, size=(h, w), scale_factor=None, mode='bilinear',
                           align_corners=self.align_corners) for out in outs]
            out = torch.cat(outs, dim=1)
            out = self.proj(out)
        else:
            out = outs[-1]

        h, w = masks[-1].shape[2:]
        masks = [F.interpolate(mask, size=(h, w), scale_factor=None, mode='bilinear',
                        align_corners=self.align_corners) for mask in masks]
        total_mask = torch.sum(torch.stack(masks, dim=0), dim=0, keepdim=False)
        wo_mask = torch.sum(torch.stack(masks[:-1], dim=0), dim=0, keepdim=False)

        kk = self.trans(kc)
        total_mask = F.interpolate(total_mask, size=outs[0].shape[2:],
                                   mode='bilinear', align_corners=False)
        pred = self.semantic_inference(kk, total_mask)

        if self.div_loss:
            inters = [F.softmax(inter.flatten(-2), -1) for inter in inters]  # b,c//2,hw
            inters = [torch.max(inter, dim=1)[0] for inter in inters]  # b, hw
            return [self.cls_seg(out) + pred, wo_mask, inters]
        return [self.cls_seg(out) + pred, wo_mask]  # [(2,150,128,128), (2,150,64,64)]

    def semantic_inference(self, mask_cls, mask_pred):
        """semantic_inference
        """
        mask_cls = F.softmax(mask_cls, dim=-1)  # 2,150,150
        mask_pred = mask_pred.sigmoid()  # 2,150,640,640
        semseg = torch.einsum("bkk,bkhw->bkhw", mask_cls, mask_pred)
        return semseg  # 2,150,640,640

    def forward_test(self, inputs):
        """forward_test
        """
        if self.mask_loss:
            return self.forward(inputs)[0]
        return self.forward(inputs)


class GhostUpscale(nn.Module):
    """GhostUpscale
    """
    def __init__(self, channels, upscale_factor, ghost_ratio=2, norm_cfg=None,
                 act_cfg=None):
        super(GhostUpscale, self).__init__()
        self.oup = channels * 2 * 2
        init_dims = math.ceil(self.oup / ghost_ratio)
        new_dims = init_dims * (ghost_ratio - 1)

        self.layers = nn.ModuleList()
        for _ in range(int(math.log2(upscale_factor))):  # 2x up each iter
            layer = nn.ModuleDict({
                'primary': ConvModule(channels, init_dims, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                'cheap': ConvModule(init_dims, new_dims, 3, 1, 1, groups=init_dims,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)
            })
            self.layers.append(layer)

    def forward(self, x):
        """forward
        """
        for layer in self.layers:
            x1 = layer['primary'](x)
            x2 = layer['cheap'](x1)
            x = torch.cat([x1, x2], dim=1)
            x = F.pixel_shuffle(x[:, :self.oup, :, :], upscale_factor=2)
        return x


class FeaTransLayer(nn.Module):
    """FeaTransLayer
    """
    def __init__(self, embed_dims=256, num_heads=4, kv_tokens=150, attn_drop_rate=0.0,
                 drop_rate=.0, qkv_bias=True, mlp_ratio=4, ln_norm_cfg=None):
        super(FeaTransLayer, self).__init__()
        _, self.norm_low = build_norm_layer(ln_norm_cfg, num_features=embed_dims)
        _, self.norm_high = build_norm_layer(ln_norm_cfg, num_features=embed_dims)
        self.cross_attn = MultiHeadAttention(embed_dims, num_heads, kv_tokens, attn_drop_rate,
                                             drop_rate, qkv_bias=qkv_bias)

        _, self.norm_mlp = build_norm_layer(ln_norm_cfg, num_features=embed_dims)
        ffn_channels = embed_dims * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dims, ffn_channels, 1, bias=True),
            nn.Conv2d(ffn_channels, ffn_channels, 3, 1, 1, groups=ffn_channels, bias=True),
            build_activation_layer(dict(type='GELU')),
            nn.Dropout(drop_rate),
            nn.Conv2d(ffn_channels, embed_dims, 1, bias=True),
            nn.Dropout(drop_rate))

    def forward(self, low, high):
        """forward
        """
        query = self.norm_low(low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # bchw
        key_value = self.norm_high(high)  # bkc
        outs = self.cross_attn(query, key_value)

        out = outs.pop('out') + low
        out = self.mlp(self.norm_mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) + out

        outs.update({'out': out})
        return outs

class MultiHeadAttention(nn.Module):
    """MultiHeadAttention
    """
    def __init__(self, embed_dims, num_heads, kv_tokens, attn_drop_rate=0., drop_rate=0.,
                 qkv_bias=True, qk_scale=None, proj_bias=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims ** -0.5

        self.q = DepthwiseSeparableConvModule(embed_dims, embed_dims, 3, 1, 1,
                                                  act_cfg=None, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, query, key_value):
        """forward
        """
        B, _, H, W = query.shape  # high resolution, a.k.a, low level

        q = self.q(query)  # 2,256,h,w
        kv = self.kv(key_value).transpose(-2, -1)
        k, v = torch.chunk(kv, chunks=2, dim=1)  # bck

        q = q.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)  # 2,4,1024,64
        k = k.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)  # 2,4,150,64
        v = v.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)  # 2,4,150,64

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # 2,4,1024,150
        attn_save = attn.clone()
        attn = torch.max(attn, -1, keepdim=True)[0].expand_as(attn) - attn  # stable training
        attn = F.softmax(attn, dim=-1)  # B(num_heads)(HW)L
        # outs = {'attn': torch.mean(attn, dim=1, keepdim=False)} # 2,1024,150 update换成=
        outs = {'attn': attn_save.sum(dim=1) / self.num_heads}  # blk
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(-2, -1).reshape(B, self.embed_dims, H, W)
        out = self.proj_drop(self.proj(out))
        outs.update({'out': out})  # 2,150,h,w

        return outs
