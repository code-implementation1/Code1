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

import collections.abc
from collections import OrderedDict
from itertools import repeat
import math
import mindspore
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import numpy as np

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=None,
                        reduce_zero_label=False):
    if label_map is None:
        label_map = dict()
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    else:
        pred_label = pred_label

    label = label

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = np.histogram(
        intersect, bins=(num_classes), range=(0, num_classes - 1))[0]
    area_pred_label = np.histogram(
        pred_label, bins=(num_classes), range=(0, num_classes - 1))[0]
    area_label = np.histogram(
        label, bins=(num_classes), range=(0, num_classes - 1))[0]
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score

def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics='mIoU',
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = np.array(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics

def pre_eval_to_metrics(pre_eval_results,
                        metrics,
                        nan_to_num=None,
                        beta=1):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics

def build_norm_layer(norm_cfg, num_features):
    if norm_cfg['type'] == "LN":
        name, module = norm_cfg['type'], nn.LayerNorm(num_features)
    elif norm_cfg['type'] == "SyncBN":
        name, module = "bn", nn.BatchNorm2d(num_features)
    return name, module

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

def build_activation_layer(act_cfg):
    if act_cfg['type'] == "ReLU":
        module = nn.ReLU(inplace=True)
    elif act_cfg['type'] == "GELU":
        module = nn.GELU()
    else:
        assert False, f"{act_cfg['type']} not support"
    return module

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
    """
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=None,
                 ffn_drop=0.,
                 add_identity=True,
                 **kwargs):
        super(FFN, self).__init__()
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out

CONV_LAYERS = {'Conv1d': nn.Conv1d,
               'Conv2d': nn.Conv2d,
               'Conv3d': nn.Conv3d,
               'Conv': nn.Conv2d}

def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    assert layer_type in CONV_LAYERS, f'Unrecognized layer type {layer_type}'

    conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)

def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def build_padding_layer(cfg, *args, **kwargs):
    if cfg['type'] == "zero":
        padding_layer = nn.ZeroPad2d
    elif cfg['type'] == "reflect":
        padding_layer = nn.ReflectionPad2d
    elif cfg['type'] == "replicate":
        padding_layer = nn.ReplicationPad2d
    else:
        assert False

    layer = padding_layer(*args, **kwargs)

    return layer

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=None,
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class TokenEmbedding_noCoord_seg(nn.Module):
    def __init__(self, num_tokens, out_dim=128, ratio=4):
        super(TokenEmbedding_noCoord_seg, self).__init__()
        length = int(math.sqrt(num_tokens) * ratio)
        length = length if length % 2 else length + 1
        pooling_size = [1, 2, 3, 6, length]
        self.pooling = nn.ModuleList([mindspore.nn.AdaptiveAvgPool2d(x) for x in pooling_size])

        self.cls_learner = nn.Linear(length*length+50, out_dim)

    def forward(self, mask):
        mask = torch.cat([pool(mask).flatten(start_dim=-2) for pool in self.pooling], -1)  # b,k,l
        out = self.cls_learner(mask)  # b,k,(h'*w') -> b,k,c

        return out  # bkc

class TokenUpdator(nn.Module):
    """Dynamic Kernel Updator in Kernel Update Head.

    Args:
        in_channels (int): The number of channels of input feature map.
            Default: 256.
        feat_channels (int): The number of middle-stage channels in
            the kernel updator. Default: 64.
        out_channels (int): The number of output channels.
        gate_sigmoid (bool): Whether use sigmoid function in gate
            mechanism. Default: True.
        gate_norm_act (bool): Whether add normalization and activation
            layer in gate mechanism. Default: False.
        activate_out: Whether add activation after gate mechanism.
            Default: False.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='LN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
    """

    def __init__(
            self,
            in_channels=256,
            feat_channels=64,
            out_channels=None,
            gate_sigmoid=True,
            gate_norm_act=False,
            activate_out=False,
            norm_cfg=None,
            act_cfg=None,
    ):
        super(TokenUpdator, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.dynamic_layer = nn.Linear(self.in_channels, self.in_channels*2)
        self.input_layer = nn.Linear(self.in_channels, self.in_channels)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature):
        parameters = self.dynamic_layer(update_feature)
        param_in, param_out = torch.chunk(parameters, chunks=2, dim=-1)

        input_feats = self.input_layer(input_feature)  # bkc -> bkc

        gate_feats = input_feats * param_in
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)

        if self.activate_out:
            param_out = self.activation(param_out)

        features = update_gate * param_out + input_feature

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features

class AdaptiveFeatureSelector_1_layers(nn.Module):
    def __init__(self, single_channels, slct_ratio=1.0, norm_cfg=None):
        super().__init__()
        self.k = int(slct_ratio * single_channels)
        self.primary = nn.Sequential(
            nn.Conv2d(single_channels, single_channels//2, kernel_size=1, stride=1, bias=False),
            build_norm_layer(norm_cfg, single_channels//2)[1],
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(single_channels//2, single_channels//2, 3, 1, 1, groups=single_channels//2),
            build_norm_layer(norm_cfg, single_channels//2)[1]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        total_channels = single_channels // 2
        self.layer_channel_slct = nn.Sequential(
            nn.Linear(total_channels, total_channels//2, bias=False),
            nn.ReLU(),
            nn.Linear(total_channels//2, total_channels, bias=False)
        )

        self.conv_out = nn.Conv2d(single_channels, self.k, 1, bias=False)

    def forward(self, x):
        x1 = self.primary(x)  # bc//2hw
        x2 = self.cheap(x1)  # bc//2hw
        layers_output = x2
        avgout = self.layer_channel_slct(self.avg_pool(layers_output).flatten(-3))   # b,c//2
        layers_output *= avgout.unsqueeze(-1).unsqueeze(-1)  # b,c//2,h,w

        # layers_output = self.merge(layers_output)

        out = torch.cat([layers_output, x1], 1)

        out = self.conv_out(out)  # b,c,h,w

        return layers_output, out  # bchw
