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
"""python export.py"""

import os
import pickle
import numpy as np
from src.evaluate import Model

import mindspore as ms
from mindspore import context

from model_utils.device_adapter import get_device_id
from model_utils.config import config


if __name__ == '__main__':
    ckpt_save_dir = config.ckpt_save_dir
    device_id = get_device_id()
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    elif config.device_target == "Ascend":
        context.set_context(device_id=device_id)

    with open(os.path.join(config.architecture_dir, 'the_best_individuals.dat'), 'rb') as f:
        individual = pickle.load(f)[0]

    net = Model(individual)
    model_parameters_dict = ms.load_checkpoint(os.path.join(config.ckpt_save_dir, 'the_best_individuals.ckpt'))
    ms.load_param_into_net(net, model_parameters_dict)

    input_arr = ms.Tensor(np.ones([1, 1, config.imageSize, config.imageSize]), ms.float32)
    ms.export(net, input_arr, file_name=config.file_name, file_format=config.file_format)
