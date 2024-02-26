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
"""network architecture"""

import copy
import math
import hashlib
import logging as logger
import numpy as np
from .Layers import ConLayer, PoolLayer, FullLayer

logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


class Individual:
    def __init__(self, input_shape, m_prob=0.2, m_eta=20.0):
        self.individual = []
        self.block_number = None
        self._m_prob = m_prob
        self._m_eta = m_eta
        self.feature_size_range = [10, 128]
        self.hidden_num_range = [1, 2000]
        self.layer_max = 20
        self.pool_layer_max = int(math.log(input_shape[0], 2))
        self.input_shape = input_shape
        self.mean = 0
        self.std = 0
        self.complexity = 0
        self.history_operation = []
        self.training_history = {}

    def __str__(self):
        message = ['length:{}, num:{}'.format(self.get_layer_size(), self.complexity),
                   'mean:{:.4f}'.format(self.mean), 'std:{:.4f}'.format(self.std)]
        for i in range(self.get_layer_size()):
            unit = self.get_layer_at(i)
            if unit.type == 1:
                message.append(str(unit))
            elif unit.type == 2:
                message.append(str(unit))
            elif unit.type == 3:
                message.append(str(unit))
            else:
                raise Exception('Incorrect unit flag')
        return ', '.join(message)

    @staticmethod
    def add_a_random_pool_layer(current_uint):
        input_size = copy.deepcopy(current_uint.input_size)
        pool_layer = PoolLayer(input_size=input_size, dropout_rate=0.2)
        return pool_layer

    @staticmethod
    def _random_select(number):
        r = np.random.random()
        for i in range(number):
            if r < (i+1.0)/number:
                return i+1
        return i+1

    @staticmethod
    def _random_int(low, high):
        if low > high:
            return np.random.randint(high, low+1)
        return np.random.randint(low, high + 1)

    @staticmethod
    def _random_float(low, high):
        return np.random.random()*(high - low) + low

    def generate_conv_layer_conv_layer(self, feature_size=None):
        if not self.individual:
            input_size = copy.deepcopy(self.input_shape)
        else:
            input_size = copy.deepcopy(self.individual[-1].output_size)
        if not feature_size:
            feature_size = self._random_int(self.feature_size_range[0], self.feature_size_range[1])
        layer_ = ConLayer(input_size=input_size, feature_size=feature_size,
                          batch_norm=np.random.random(),
                          active_function=np.random.random(),
                          order=np.random.random(),
                          initializer=np.random.random(),
                          dropout_rate=0.2)
        return layer_

    def generate_pool_layer(self):
        input_size = copy.deepcopy(self.individual[-1].output_size)
        layer_ = PoolLayer(input_size, dropout_rate=0.2)
        return layer_

    def generate_full_layer(self, hidden_num=None):
        input_size = copy.deepcopy(self.individual[-1].output_size)
        if not hidden_num:
            hidden_num = self._random_int(self.hidden_num_range[0], self.hidden_num_range[1])
        layer_ = FullLayer(input_size=input_size, hidden_num=hidden_num,
                           batch_norm=np.random.random(),
                           active_function=np.random.random(),
                           order=np.random.random(),
                           initializer=np.random.random(),
                           dropout_rate=0.2)
        return layer_

    def initialize_min(self):
        self.individual.append(self.generate_conv_layer_conv_layer())
        self.individual.append(self.generate_pool_layer())
        self.individual.append(self.generate_full_layer(hidden_num=2))
        self.block_number = self._get_pool_number()

    def initialize_random(self):
        pool_layer_size = self._random_int(1, self.pool_layer_max)
        conv_layer_size = self._random_int(1, self.layer_max - pool_layer_size - 1)
        full_layer_size = self._random_int(1, self.layer_max - pool_layer_size - conv_layer_size)
        layer_index_list = [0 for _ in range(conv_layer_size - 1)]
        layer_index_list.extend([1 for _ in range(pool_layer_size)])
        np.random.shuffle(layer_index_list)
        layer_index_list.insert(0, 0)
        layer_index_list.extend([2 for _ in range(full_layer_size)])
        self.individual.append(self.generate_conv_layer_conv_layer())
        for index in range(1, len(layer_index_list) - 1):
            if layer_index_list[index] == 0:
                self.individual.append(self.generate_conv_layer_conv_layer())
            elif layer_index_list[index] == 1:
                self.individual.append(self.generate_pool_layer())
            elif layer_index_list[index] == 2:
                self.individual.append(self.generate_full_layer())
        self.individual.append(self.generate_full_layer(2))
        self.block_number = self._get_pool_number()

    def initialize_random_2(self):
        conv_num = self._random_int(2, 3)
        pool_num = 1
        full_num = self._random_int(2, 3)
        layer_index_list = [1 for _ in range(conv_num - 1)]
        layer_index_list.extend([2 for _ in range(pool_num)])
        np.random.shuffle(layer_index_list)
        layer_index_list.insert(0, 1)
        layer_index_list.extend([3 for _ in range(full_num)])
        self.individual.append(self.generate_conv_layer_conv_layer())
        for index in range(1, len(layer_index_list) - 1):
            if layer_index_list[index] == 1:
                self.individual.append(self.generate_conv_layer_conv_layer())
            elif layer_index_list[index] == 2:
                self.individual.append(self.generate_pool_layer())
            else:
                self.individual.append(self.generate_full_layer())
        self.individual.append(self.generate_full_layer(2))
        self.block_number = self._get_pool_number()

    def clear_performance_score(self):
        self.mean = 0
        self.std = 0
        self.complexity = 0

    def is_del_cur_layer(self, cur_index):
        cur_uint = self.get_layer_at(cur_index)
        if cur_uint.type == 1:
            if self._get_conv_number() == 1:
                return False
            if cur_index == 0:
                return False
        elif cur_uint.type == 2:
            if self._get_pool_number() == 1:
                return False
        else:
            if self._get_full_number() == 1:
                return False
        return True

    def set_performance_score(self, mean, std, complexity):
        logger.info('%f-%f-%d', mean, std, complexity)
        self.mean = mean
        self.std = std
        self.complexity = complexity

    def set_layers(self, layer_list):
        self.clear_performance_score()
        self.individual = layer_list
        self.block_number = self._get_pool_number()

    def add_a_random_conv_layer(self, current_uint):
        input_shape = copy.deepcopy(current_uint.input_size)
        if current_uint.type == 3:
            feature_size = input_shape[-1]
        elif current_uint.type == 1:
            last_feature_size = current_uint.input_size[-1]
            next_feature_size = current_uint.output_size[-1]
            feature_size = self._random_int(last_feature_size, next_feature_size)
        else:
            feature_size = current_uint.input_size[-1]
            input_shape = copy.deepcopy(current_uint.input_size)
        conv_layer = ConLayer(input_size=input_shape, feature_size=feature_size,
                              batch_norm=np.random.random(),
                              active_function=np.random.random(),
                              order=np.random.random(),
                              initializer=np.random.random(),
                              dropout_rate=0.2)
        return conv_layer

    def add_a_random_full_layer(self, current_uint):
        input_size = current_uint.input_size
        if isinstance(input_size, int):
            last_number = input_size
        else:
            last_number = input_size[0]*input_size[1]*input_size[2]
            last_number = min(self.hidden_num_range[1], max(self.hidden_num_range[0], last_number))
        next_number = current_uint.hidden_num
        hidden_number = self._random_int(next_number, last_number)
        full_layer = FullLayer(input_size, hidden_num=hidden_number,
                               batch_norm=np.random.random(),
                               active_function=np.random.random(),
                               order=np.random.random(),
                               initializer=np.random.random(),
                               dropout_rate=0.2)
        return full_layer

    def check_consistency(self):
        last_layer = self.get_layer_at(0)
        if not last_layer.input_size == self.input_shape:
            last_layer.input_size = copy.deepcopy(self.input_shape)
            last_layer.feature_size = self.input_shape[-1]*2
            last_layer.output_size = [self.input_shape[0], self.input_shape[1], last_layer.feature_size]
        for current_layer in self.individual[1:]:
            if not last_layer.output_size == current_layer.input_size:
                if current_layer.type == 3:
                    current_layer.input_size = copy.deepcopy(last_layer.output_size)
                elif current_layer.type == 1:
                    current_layer.input_size = copy.deepcopy(last_layer.output_size)
                    current_layer.output_size = [current_layer.input_size[0],
                                                 current_layer.input_size[1],
                                                 current_layer.feature_size]
                elif current_layer.type == 2:
                    current_layer.input_size = copy.deepcopy(last_layer.output_size)
                    current_layer.output_size = [int(current_layer.input_size[0]/2),
                                                 int(current_layer.input_size[1]/2),
                                                 current_layer.input_size[2]]
            last_layer = current_layer
        self.block_number = self._get_pool_number()

    def mutation(self):
        if np.random.random() < self._m_prob:
            for cur_index in range(self.get_layer_size() - 1, -1, -1):
                if np.random.random() < 0.5:
                    op = self._random_select(3)
                    if op == 1:
                        cur_length = self.get_layer_size()
                        if cur_length < self.layer_max:
                            self.generate_a_new_layer(cur_index)
                        else:
                            self.mutation_a_unit(cur_index)
                    elif op == 2:
                        if cur_index == self.get_layer_size() - 1:
                            continue
                        self.mutation_a_unit(cur_index)
                    else:
                        if cur_index == self.get_layer_size() - 1:
                            continue
                        if self.is_del_cur_layer(cur_index):
                            del self.individual[cur_index]
                            self.check_consistency()
        self.check_consistency()

    def pm(self, xl, xu, x):
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        r = np.random.random()
        mut_pow = 1.0 / (self._m_eta + 1.)
        if r < 0.5:
            xy = 1.0 - delta_1
            val = 2.0 * r + (1.0 - 2.0 * r) * xy**(self._m_eta + 1)
            delta_q = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta_2
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * xy**(self._m_eta + 1)
            delta_q = 1.0 - val**mut_pow
        x = x + delta_q * (xu - xl)
        x = min(max(x, xl), xu)
        return x

    def mutation_a_unit(self, cur_index):
        cur_uint = self.get_layer_at(cur_index)
        if cur_uint.type == 1:
            feature_s = cur_uint.feature_size
            feature_s = int(self.pm(self.feature_size_range[0], self.feature_size_range[1], feature_s))
            cur_uint.feature_size = feature_s
            batch_norm = cur_uint.batch_norm
            batch_norm = self.pm(0.0, 1.0, batch_norm)
            cur_uint.batch_norm = batch_norm
            # active function
            active_function = cur_uint.active_function
            active_function = self.pm(0.0, 1.0, active_function)
            cur_uint.active_function = active_function
            # order
            order = cur_uint.order
            order = self.pm(0.0, 1.0, order)
            cur_uint.order = order
            # initializer
            initializer = cur_uint.initializer
            initializer = self.pm(0.0, 1.0, initializer)
            cur_uint.initializer = initializer
            # dropout
            dropout_rate = cur_uint.dropout_rate
            dropout_rate = self.pm(0.0, 1.0, dropout_rate)
            cur_uint.dropout_rate = dropout_rate
        elif cur_uint.type == 2:
            # dropout
            dropout_rate = cur_uint.dropout_rate
            dropout_rate = self.pm(0.0, 1.0, dropout_rate)
            cur_uint.dropout_rate = dropout_rate
        else:
            if not cur_index == self.get_layer_size() - 1:
                hidden_n = cur_uint.hidden_num
                hidden_n = int(self.pm(self.hidden_num_range[0], self.hidden_num_range[1], hidden_n))
                cur_uint.hidden_num = hidden_n
            # batch norm
            batch_norm = cur_uint.batch_norm
            batch_norm = self.pm(0.0, 1.0, batch_norm)
            cur_uint.batch_norm = batch_norm
            # active function
            active_function = cur_uint.active_function
            active_function = self.pm(0.0, 1.0, active_function)
            cur_uint.active_function = active_function
            # order
            order = cur_uint.order
            order = self.pm(0.0, 1.0, order)
            cur_uint.order = order
            # initializer
            initializer = cur_uint.initializer
            initializer = self.pm(0.0, 1.0, initializer)
            cur_uint.initializer = initializer
            # dropout
            dropout_rate = cur_uint.dropout_rate
            dropout_rate = self.pm(0.0, 1.0, dropout_rate)
            cur_uint.dropout_rate = dropout_rate
        self.check_consistency()

    def get_layer_size(self):
        return len(self.individual)

    def get_layer_at(self, index):
        return self.individual[index]

    def generate_a_new_layer(self, cur_index):
        cur_uint = self.get_layer_at(cur_index)
        if cur_index == 0:
            new_layer = self.add_a_random_conv_layer(cur_uint)
        else:
            if cur_uint.type == 3:
                if isinstance(cur_uint.input_size, int):
                    add_layer_op = 1
                elif self._is_add_pool_layer():
                    add_layer_op = self._random_select(3)
                else:
                    add_layer_op = self._random_select(2)
                if add_layer_op == 1:
                    new_layer = self.add_a_random_full_layer(cur_uint)
                elif add_layer_op == 2:
                    new_layer = self.add_a_random_conv_layer(cur_uint)
                else:
                    new_layer = self.add_a_random_pool_layer(cur_uint)
            else:
                if self._is_add_pool_layer():
                    add_layer_op = self._random_select(2)
                else:
                    add_layer_op = 1
                if add_layer_op == 1:
                    new_layer = self.add_a_random_conv_layer(cur_uint)
                else:
                    new_layer = self.add_a_random_pool_layer(cur_uint)
        self.individual.insert(cur_index, new_layer)
        self.check_consistency()

    def get_md5(self):
        message = []
        for i in range(self.get_layer_size()):
            unit = self.get_layer_at(i)
            if unit.type == 1:
                message.append(str(unit))
            elif unit.type == 2:
                message.append(str(unit))
            elif unit.type == 3:
                message.append(str(unit))
            else:
                raise Exception('Incorrect unit flag')
        message = ','.join(message)
        md5 = hashlib.md5()
        md5.update(message.encode('utf-8'))
        return md5.hexdigest()

    def _get_conv_number(self):
        conv_number = 0
        for layer in self.individual:
            if layer.type == 1:
                conv_number += 1
        return conv_number

    def _get_pool_number(self):
        pool_number = 0
        for layer in self.individual:
            if layer.type == 2:
                pool_number += 1
        return pool_number

    def _get_full_number(self):
        full_number = 0
        for layer in self.individual:
            if layer.type == 3:
                full_number += 1
        return full_number

    def _is_add_pool_layer(self):
        if self._get_pool_number() >= self.pool_layer_max:
            return False
        return True
