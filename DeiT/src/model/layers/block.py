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
#
# This file has been derived from the https://github.com/facebookresearch/deit
# repository and modified.
# ============================================================================

"""The implementation of Block layer"""

from functools import partial

import mindspore.nn as nn

from .custom_identity import CustomIdentity
from .attention import Attention
from .mlp import Mlp
from .drop_path import DropPath


class Block(nn.Cell):
    """
    The Block layer
    The Pytorch implementation can be found by this link:
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L240
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=partial(nn.GELU, approximate=False),
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else CustomIdentity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
