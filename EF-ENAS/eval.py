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
"""python eval.py"""

import os
import pickle
import logging as logger
import numpy as np
from src.evaluate import Model
from data.get_data import get_ri_for_evaluate

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size

from model_utils.device_adapter import get_device_id
from model_utils.config import config

logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


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

    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=config.group_size)

    with open(os.path.join(config.architecture_dir, 'the_best_individuals.dat'), 'rb') as f:
        individual = pickle.load(f)[0]

    net = Model(individual)
    train_set, valid_set = get_ri_for_evaluate(config.data_path, batch_size=config.batch_size)

    parameters, loss_scale_manager = 0, None
    for p in net.trainable_params():
        parameters += np.cumprod(p.shape)[-1]

    opt = nn.Adam(net.trainable_params(), learning_rate=config.learning_rate)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    model = ms.Model(net, loss_fn=loss, optimizer=opt, metrics={'acc', 'loss'},
                     amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=loss_scale_manager)

    model_parameters_dict = ms.load_checkpoint(os.path.join(config.ckpt_save_dir, 'the_best_individuals.ckpt'))
    ms.load_param_into_net(net, model_parameters_dict)

    batch_num = train_set.get_dataset_size()
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor()

    cbs = [time_cb, loss_cb]
    result_dict = model.eval(valid_set)

    result_dict['parameters'] = parameters
    logger.info(result_dict)
