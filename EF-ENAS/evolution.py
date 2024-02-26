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
"""python evolution.py"""

import os
import copy
import pickle
from time import gmtime
from time import strftime
from shutil import copyfile
import stat
import logging as logger
import numpy as np
from src.evaluate import Evaluate
from src.Population import Population
from src.check_networks import is_same_structure
from src.individual_set_dict import IndividualHistory
from model_utils.config import config

logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class EvolutionCNN:
    def __init__(self, population_size, total_generation_num, input_shape, epochs,
                 batch_size, train_data_len, test_data_len):
        self.current_gen_no = -1
        self.total_generation_num = total_generation_num
        self._elite = 0.2
        self.population_size = population_size
        self.populations = None
        self._input_shape = input_shape
        self._epochs = epochs
        self._batch_size = batch_size
        self._train_data_length = train_data_len
        self._test_data_length = test_data_len
        self.save_cache_path = config.output_path
        if not os.path.exists(self.save_cache_path):
            os.makedirs(self.save_cache_path)
        self.save_offspring_path = config.offspring_path
        if not os.path.exists(self.save_offspring_path):
            os.mkdir(self.save_offspring_path)

    @staticmethod
    def _select_mean(mean_1, mean_2):
        mean_threshold = 0.05
        if abs(mean_1 - mean_2) < mean_threshold:
            return 0
        if mean_1 > mean_2:
            value = 1
        else:
            value = -1
        return value

    @staticmethod
    def _select_complexity(com_1, com_2):
        com_max = max(com_1, com_2)
        com_min = min(com_1, com_2)
        complexity_threshold = 10.0
        if com_max/com_min < complexity_threshold:
            value = 0
        elif com_1 < com_2:
            value = 1
        else:
            value = -1
        return value

    @staticmethod
    def _select_generalize(layer_1, layer_2):
        generalize_threshold = 0.1
        layer_1_max = 0.0
        layer_2_max = 0.0
        for i in range(len(layer_1) - 1):
            if not layer_1[i].type == 1:
                continue
            if layer_1[i].dropout_rate > layer_1_max:
                layer_1_max = layer_1[i].dropout_rate
        for i in range(len(layer_2) - 1):
            if not layer_2[i].type == 1:
                continue
            if layer_2[i].dropout_rate > layer_2_max:
                layer_2_max = layer_2[i].dropout_rate
        if abs(layer_1_max - layer_2_max) < generalize_threshold:
            value = 0
        elif layer_1_max > layer_2_max:
            value = 1
        else:
            value = -1
        return value

    def initialize_population(self):
        logger.info('initializing populations with number %d ...', self.population_size)
        self.populations = Population(self._input_shape, self.population_size)
        self.populations.print_block()
        self.current_gen_no += 1

    def save_populations(self):
        message = {'gen_no': self.current_gen_no, 'pops': self.populations,
                   'create_time': strftime('%Y-%m-%d %H:%M:%S', gmtime())}
        save_path = os.path.join(self.save_cache_path, 'gen_{:03d}'.format(self.current_gen_no))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR

        fd = os.open(os.path.join(self.save_cache_path, 'pops.dat'), flags, modes)

        with os.fdopen(fd, 'wb') as f:
            pickle.dump(message, f)
        copyfile(os.path.join(self.save_cache_path, 'pops.dat'),
                 os.path.join(save_path, 'pops.dat'))

        fd = os.open(os.path.join(save_path, 'pops.txt'), flags, modes)
        with os.fdopen(fd, 'w') as f:
            for individual in self.populations.populations:
                f.write(str(individual))
                f.write('\n')

    def load_populations(self):
        if not os.path.exists(os.path.join(self.save_cache_path, 'pops.dat')):
            return False
        with open(os.path.join(self.save_cache_path, 'pops.dat'), 'rb') as f:
            data = pickle.load(f)
        self.current_gen_no = data['gen_no']
        self.populations = data['pops']
        self.populations.save_history()
        self.populations.individual_history.load_individual_history()
        return True

    def save_offspring_populations(self, offspring_populations):
        message = {'gen_no': self.current_gen_no, 'pops': offspring_populations,
                   'create_time': strftime('%Y-%m-%d %H:%H:%M:%S', gmtime())}
        save_path = os.path.join(self.save_offspring_path, 'gen_{:03d}'.format(self.current_gen_no))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR

        fd = os.open(os.path.join(save_path, 'new_offspring.dat'), flags, modes)
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(message, f)

        fd = os.open(os.path.join(save_path, 'new_offspring.txt'), flags, modes)
        with os.fdopen(fd, 'w') as f:
            for offspring in offspring_populations.populations:
                f.write(str(offspring))
                f.write('\n')

    def evaluate_fitness(self):
        logger.info('evaluating fitness')
        evaluate = Evaluate(self.populations, self._input_shape, self._epochs, self._batch_size,
                            self._train_data_length, self._test_data_length)
        evaluate.parse_populations(self.current_gen_no)
        self.save_populations()
        self.populations.save_history()

    def evolution(self):
        logger.info('mutation and crossover ...')
        self.current_gen_no += 1
        offspring_list = []
        individual_history_copy = IndividualHistory()
        for _ in range(int(self.populations.get_populations_size()/2)):
            individual_1, individual_2 = self.tournament_select()
            individual_1, individual_2 = self.populations.crossover_base_cell(individual_1, individual_2)
            ind_1 = copy.deepcopy(individual_1)
            ind_1.mutation()
            while individual_history_copy.check_id(ind_1.get_md5()):
                ind_1 = copy.deepcopy(individual_1)
                ind_1.mutation()
            ind_2 = copy.deepcopy(individual_2)
            ind_2.mutation()
            while individual_history_copy.check_id(ind_2.get_md5()):
                ind_2 = copy.deepcopy(individual_2)
                ind_2.mutation()
            offspring_list.append(ind_1)
            offspring_list.append(ind_2)
            individual_history_copy.add_id(ind_1.get_md5(), 0)
            individual_history_copy.add_id(ind_2.get_md5(), 0)
        offspring_pops = Population(self._input_shape, 0)
        offspring_pops.set_populations(offspring_list)
        evaluate = Evaluate(offspring_pops, self._input_shape, self._epochs, self._batch_size,
                            self._train_data_length, self._test_data_length)
        evaluate.parse_populations(self.current_gen_no)
        self.save_offspring_populations(offspring_pops)
        offspring_pops.save_history(save_to_disk=False)
        self.populations.individual_history.extend_id_list(offspring_pops.individual_history.individual_dict)
        self.populations.save_history()
        self.populations.extend_populations(offspring_pops.populations)

    def environment_select(self):
        assert self.populations.get_populations_size() == 2*self.population_size
        e_count = int(np.floor(self.population_size*self._elite/2)*2)
        individual_list = self.populations.populations
        individual_list.sort(key=lambda x: x.mean, reverse=True)
        elite_list = individual_list[:e_count]
        left_list = individual_list[e_count:]
        for _ in range(self.population_size - e_count):
            winner, _ = self.tournament_select(left_list)
            elite_list.append(winner)
            left_list_len = len(left_list)
            for individual_index in range(left_list_len):
                if winner.get_md5() == left_list[individual_index].get_md5():
                    left_list.pop(individual_index)
                    break
            elite_list.sort(key=lambda x: x.mean, reverse=True)
        self.populations.set_populations(elite_list)
        self.save_populations()

    def tournament_select(self, populations_raw=None):
        if not populations_raw:
            populations = copy.deepcopy(self.populations.populations)
        else:
            populations = copy.deepcopy(populations_raw)
        populations_list = []
        while populations:
            same_list = []
            individual = populations.pop()
            same_list.append(individual)
            for individual_index in range(len(populations)-1, -1, -1):
                if is_same_structure(individual, populations[individual_index]):
                    same_list.append(populations.pop(individual_index))
            populations_list.append(same_list)
        less_list = []
        for i in range(len(populations_list)-1, -1, -1):
            if len(populations_list[i]) < 3:
                less_list.extend(populations_list.pop(i))
        if not less_list:
            populations_list.append(less_list)
        if np.random.random() < 0.5:
            pop_index = np.random.randint(0, len(populations_list))
            pop = populations_list[pop_index]
        else:
            if populations_raw:
                pop = populations_raw
            else:
                pop = self.populations.populations
        ind1_index = np.random.randint(0, len(pop))
        ind2_index = np.random.randint(0, len(pop))
        ind3_index = np.random.randint(0, len(pop))
        winter_1 = self.selection_(pop[ind1_index], pop[ind2_index])
        winter_2 = self.selection_(pop[ind3_index], winter_1)

        ind1_index = np.random.randint(0, len(pop))
        ind2_index = np.random.randint(0, len(pop))
        ind3_index = np.random.randint(0, len(pop))
        winter_3 = self.selection_(pop[ind1_index], pop[ind2_index])
        winter_4 = self.selection_(pop[ind3_index], winter_3)
        return copy.deepcopy(winter_2), copy.deepcopy(winter_4)

    def selection_(self, ind1, ind2):
        mean_order = self._select_mean(ind1.mean, ind2.mean)
        if mean_order == 0:
            complexity_order = self._select_complexity(ind1.complexity, ind2.complexity)
            if complexity_order == 0:
                generalize_order = self._select_generalize(ind1.individual, ind2.individual)
                if generalize_order == 0:
                    if ind1.mean > ind2.mean:
                        value = ind1
                    else:
                        value = ind2
                elif generalize_order == 1:
                    value = ind1
                else:
                    value = ind2
            elif complexity_order == 1:
                value = ind1
            else:
                value = ind2
        elif mean_order == 1:
            value = ind1
        else:
            value = ind2
        return value

    def start_evolution(self):
        self.initialize_population()
        self.evaluate_fitness()
        for current_gen_no in range(self.total_generation_num):
            logger.info('%d/%d generation', current_gen_no, self.total_generation_num)
            self.evolution()
            self.environment_select()

    def restart_evolution(self):
        if not self.load_populations():
            self.initialize_population()
            self.evaluate_fitness()
        for current_gen_no in range(self.current_gen_no, self.total_generation_num):
            logger.info('%d/%d generation', current_gen_no, self.total_generation_num)
            self.evolution()
            self.environment_select()


if __name__ == '__main__':
    cnn = EvolutionCNN(population_size=100, total_generation_num=50, input_shape=[28, 28, 1],
                       epochs=config.epoch_size, batch_size=config.batch_size, train_data_len=5000, test_data_len=1000)
    cnn.start_evolution()
    print('evo finished.')
