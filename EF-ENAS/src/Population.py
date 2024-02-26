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
"""network population"""

import logging as logger
import numpy as np
from src.Individual import Individual
from src.individual_set_dict import IndividualHistory

logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


class Population:
    def __init__(self, input_size, pops_num, x_prob=0.9, x_eta=20):
        self.data_shape = input_size
        self._x_prob = x_prob
        self._x_eta = x_eta
        self._current_pops_num = pops_num
        self.populations = []
        self.individual_history = IndividualHistory()
        for _ in range(pops_num):
            individual = Individual(self.data_shape)
            individual.initialize_random_2()
            self.populations.append(individual)

    def __str__(self):
        message = []
        for index in range(self.get_populations_size()):
            message.append(str(self.get_populations_at(index)))
        return '\n'.join(message)

    @staticmethod
    def _get_cell(individual):
        cell_list = []
        cell = [individual.individual[0]]
        for i in range(1, len(individual.individual)):
            if individual.individual[i].type == cell[-1].type:
                cell.append(individual.individual[i])
            else:
                if len(cell) == 1:
                    cell.append(individual.individual[i])
                    cell_list.append(cell)
                    cell = [individual.individual[i]]
                else:
                    cell_list.append(cell)
                    cell = [cell[-1], individual.individual[i]]
                    cell_list.append(cell)
                    cell = [individual.individual[i]]
        cell_list.append(cell)
        return cell_list

    @staticmethod
    def _is_same_cell(cell_1, cell_2):
        if len(cell_1) == len(cell_2):
            for item_1, item_2 in zip(cell_1, cell_2):
                if not item_1.type == item_2.type:
                    return False
            return True
        return False

    @staticmethod
    def _is_crossover(cell_list_1, cell_list_2):
        cell_1 = []
        cell_2 = []
        if cell_list_1 == [] or cell_list_2 == []:
            return False

        for item in cell_list_1:
            tmp = 0
            if item[0].type == 3:
                tmp = 1
            if item[-1].type == 3:
                tmp += 1
            cell_1.append(tmp)
        for item in cell_list_2:
            tmp = 0
            if item[0].type == 3:
                tmp = 1
            if item[-1].type == 3:
                tmp += 1
            cell_2.append(tmp)
        cell_1_index = list(range(len(cell_1)))
        np.random.shuffle(cell_1_index)
        cell_2_index = list(range(len(cell_2)))
        np.random.shuffle(cell_2_index)
        for item_1 in cell_1_index:
            for item_2 in cell_2_index:
                if cell_1[item_1] == cell_2[item_2]:
                    return cell_list_1[item_1], cell_list_2[item_2]
        return False

    def save_history(self, save_to_disk=True):
        self.individual_history.individual_dict = {}
        for individual in self.populations:
            self.individual_history.add_id(individual.get_md5(), individual.mean)
        if save_to_disk:
            self.individual_history.save_individual_history()

    def print_block(self):
        for individual in self.populations:
            layer_message = 'len:{:02d}'.format(len(individual.individual))
            for layer in individual.individual:
                if layer.type == 1:
                    layer_message += ', conv'
                elif layer.type == 2:
                    layer_message += ', pool'
                else:
                    layer_message += ', full'
            logger.info(layer_message)

    def set_populations(self, new_pops):
        self.populations = new_pops
        self._current_pops_num = len(new_pops)

    def extend_populations(self, new_pops):
        self.populations.extend(new_pops)
        self._current_pops_num = len(self.populations)

    def get_populations_size(self):
        return self._current_pops_num

    def get_populations_at(self, population_index):
        return self.populations[population_index]

    def crossover_base_cell(self, ind1, ind2):
        ind1_cell = self._get_cell(ind1)
        ind2_cell = self._get_cell(ind2)
        ind1_tmp = [ind1_cell[0], ind1_cell[-1]]
        ind2_tmp = [ind2_cell[0], ind2_cell[-1]]
        for i in range(len(ind1_cell)-1, -1, -1):
            for j in range(len(ind2_cell)-1, -1, -1):
                cell_1 = ind1_cell[i]
                cell_2 = ind2_cell[j]
                if self._is_same_cell(cell_1, cell_2):
                    ind1_cell.pop(i)
                    ind2_cell.pop(j)
                    break
        crossover_cells = self._is_crossover(ind1_cell, ind2_cell)
        if not crossover_cells:
            return ind2, ind1
        crossover_cell_1 = crossover_cells[0]
        crossover_cell_2 = crossover_cells[1]
        if id(crossover_cell_1[0]) == id(ind1.individual[0]):
            crossover_cell_2 = ind2_tmp[0]
        elif id(crossover_cell_2[0]) == id(ind2.individual[0]):
            crossover_cell_1 = ind1_tmp[0]
        # p1
        first = 0
        second = 0
        for i in range(len(ind1.individual)):
            if id(ind1.individual[i]) == id(crossover_cell_1[0]):
                first = i
            if id(ind1.individual[i]) == id(crossover_cell_1[-1]):
                second = i+1
                break
        p1_layer_list = ind1.individual[:first]
        p1_layer_list.extend(crossover_cell_2)
        p1_layer_list.extend(ind1.individual[second:])
        # p2
        for i in range(len(ind2.individual)):
            if id(ind2.individual[i]) == id(crossover_cell_2[0]):
                first = i
            if id(ind2.individual[i]) == id(crossover_cell_2[-1]):
                second = i+1
        p2_layer_list = ind2.individual[:first]
        p2_layer_list.extend(crossover_cell_1)
        p2_layer_list.extend(ind2.individual[second:])
        ind1.set_layers(p1_layer_list)
        ind2.set_layers(p2_layer_list)
        ind1.check_consistency()
        ind2.check_consistency()
        return ind1, ind2
