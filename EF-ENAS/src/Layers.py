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
"""network layer"""

import math
import mindspore.nn as nn
from mindspore import ops
from mindspore.nn import ReLU, Tanh, ELU


class CReLU(nn.Cell):
    def __init__(self):
        super(CReLU, self).__init__()
        self.relu = nn.ReLU()
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        x1 = self.relu(x)
        x2 = self.relu(-x)
        out = self.concat((x1, x2))

        return out


class ConLayer:
    def __init__(self, input_size, feature_size, batch_norm, active_function, order, initializer, dropout_rate):
        self.input_size = input_size
        self.output_size = [input_size[0], input_size[1], feature_size]
        self.filter_size = (3, 3)
        self.feature_size = feature_size
        self.batch_norm = batch_norm
        self.active_function = active_function
        self.order = order
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.score = 0.0
        self.type = 1

    def __str__(self):
        if self.order > 0.5:
            message = 'C[num:{}, BN:{}, fn:{}, init:{}, dp:{}]'.format(
                self.feature_size, self.get_bn(), self.get_active_fn_name(), self.get_initializer(),
                self.get_dropout_rate())
        else:
            message = 'C[num:{}, fn:{}, BN:{}, init:{}, dp:{}]'.format(
                self.feature_size, self.get_active_fn_name(), self.get_bn(), self.get_initializer(),
                self.get_dropout_rate())
        return message

    def get_bn(self):
        if self.batch_norm > 0.5:
            return True
        return False

    def get_active_fn_name(self):
        if self.active_function < 0.25:
            function = 'crelu'
        elif self.active_function < 0.5:
            function = 'relu'
        elif self.active_function < 0.75:
            function = 'elu'
        else:
            function = 'tanh'
        return function

    def get_active_fn(self):
        if self.active_function < 0.25:
            function = CReLU
        elif self.active_function < 0.5:
            function = ReLU
        elif self.active_function < 0.75:
            function = ELU
        else:
            function = Tanh
        return function

    def get_initializer(self):
        return 'xavier_uniform' if self.initializer < 0.5 else 'he_uniform'

    def get_dropout_rate(self):
        return math.floor(self.dropout_rate*100)/100


class PoolLayer:
    def __init__(self, input_size, dropout_rate):
        self.input_size = input_size
        self.output_size = [int(input_size[0]/2), int(input_size[1]/2), input_size[2]]
        self.kernel_size = (2, 2)
        self.dropout_rate = dropout_rate
        self.score = 0.0
        self.type = 2

    def __str__(self):
        message = 'P[dp:{}]'.format(self.get_dropout_rate())
        return message

    def get_dropout_rate(self):
        return math.floor(self.dropout_rate*100)/100


class FullLayer:
    def __init__(self, input_size, hidden_num, batch_norm, active_function, order, initializer, dropout_rate):
        self.input_size = input_size
        self.output_size = hidden_num
        self.hidden_num = hidden_num
        self.batch_norm = batch_norm
        self.active_function = active_function
        self.order = order
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.score = 0.0
        self.type = 3

    def __str__(self):
        if self.order > 0.5:
            message = 'F[num:{}, BN:{}, fn:{}, init:{}, dp:{}]'.format(
                self.hidden_num, self.get_bn(), self.get_active_fn_name(), self.get_initializer(),
                self.get_dropout_rate()
            )
        else:
            message = 'F[num:{}, fn:{}, BN:{}, init:{}, dp:{}]'.format(
                self.hidden_num, self.get_active_fn_name(), self.get_bn(), self.get_initializer(),
                self.get_dropout_rate()
            )
        return message

    def get_bn(self):
        if self.batch_norm > 0.5:
            return True
        return False

    def get_active_fn_name(self):
        if self.active_function < 0.25:
            function = 'crelu'
        elif self.active_function < 0.5:
            function = 'relu'
        elif self.active_function < 0.75:
            function = 'elu'
        else:
            function = 'tanh'
        return function

    def get_active_fn(self):
        if self.active_function < 0.25:
            function = CReLU
        elif self.active_function < 0.5:
            function = ReLU
        elif self.active_function < 0.75:
            function = ELU
        else:
            function = Tanh
        return function

    def get_initializer(self):
        return 'xavier_uniform' if self.initializer < 0.5 else 'he_uniform'

    def get_dropout_rate(self):
        return math.floor(self.dropout_rate*100)/100
