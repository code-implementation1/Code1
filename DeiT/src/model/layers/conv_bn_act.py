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
# This file has been derived from the https://github.com/huggingface/pytorch-image-models
# repository and modified.
# ============================================================================

"""Patch embedding layer """

import mindspore.nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import convert_norm_act_type


class ConvBnAct(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=nn.ReLU, apply_act=True,
                 drop_block=None, aa_layer=None):
        super(ConvBnAct, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, group=groups, has_bias=False)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x
