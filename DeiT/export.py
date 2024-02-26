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
"""
Script for export DeiT to ONNX or MINDIR format.
"""
import argparse
from functools import partial

import numpy as np
import mindspore as ms
import mindspore.nn as nn

from src.model.deit import create_model


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Export DeiT model')
    parser.add_argument('--model', default='deit_base_patch16_224', type=str,
                        choices=[
                            'deit_base_patch16_224',
                            'deit_base_distilled_patch16_224',
                            'deit_tiny_patch16_224',
                            'deit_small_patch16_224',
                            'deit_tiny_distilled_patch16_224',
                            'deit_small_distilled_patch16_224',
                            'deit_base_patch16_384',
                            'deit_base_distilled_patch16_384'
                        ],
                        help='Name of the model')

    parser.add_argument('--checkpoint-path', type=str,
                        required=True, help='Checkpoint file path')

    parser.add_argument('--device-target', type=str, default='GPU',
                        choices=['CPU', 'GPU', 'Ascend'],
                        help='run device_target')

    parser.add_argument('--file-format', type=str, default='MINDIR',
                        choices=['ONNX', 'MINDIR', 'AIR'],
                        help='file format')

    parser.add_argument('--fix-gelu', action='store_true',
                        help='file format')

    return parser.parse_args()


class NetWithSoftmax(nn.Cell):
    """Network with softmax at the end."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.softmax = nn.Softmax()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return self.softmax(self.net(x))


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)

    try:
        if args.fix_gelu:
            act_mlp_layer = partial(nn.GELU, approximate=True)
        else:
            act_mlp_layer = partial(nn.GELU, approximate=False)

        model = create_model(
            model_name=args.model, checkpoint_path=args.checkpoint_path,
            act_mlp_layer=act_mlp_layer
        )
        model.set_train(False)

        input_shape = [
            1, model.default_cfg['in_chans'],
            model.default_cfg['img_size'], model.default_cfg['img_size']
        ]

        input_array = ms.Tensor(
            np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        )
        model = NetWithSoftmax(model)

        ms.export(
            model, input_array,
            file_name=f'{args.model}',
            file_format=args.file_format
        )
    except RuntimeError:
        print('Currently mindspore does not support layer GELU with '
              '`approximate=False`. You can change it to True via flag '
              '`--fix-gelu`, but it can affect to the quality of the model. '
              'Please read README.md for more info.')


if __name__ == '__main__':
    main()
