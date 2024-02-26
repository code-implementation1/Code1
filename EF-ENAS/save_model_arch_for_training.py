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
"""python save_model_arch_for_training.py"""

import os
import pickle
import stat
import logging as logger
from model_utils.config import config

logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


if __name__ == '__main__':
    the_best_individual = []
    cache_path = config.output_path
    cache_path = os.path.abspath(cache_path)
    logger.info(os.listdir(cache_path))
    populations_list = []
    for i in range(51):
        path = os.path.join(cache_path, 'gen_{:03d}'.format(i), 'pops.dat')
        with open(path, 'rb') as f:
            data = pickle.load(f)['pops'].populations
            populations_list.append(data)
    the_new_population = populations_list[-1][:10]
    the_new_block, max_index, max_dropout = [], [], 0
    for index_individual, individual in enumerate(the_new_population):
        block, individual_max_dropout = [], 0
        for item in individual.individual:
            if item.type == 1:
                block.append('conv[{:.2f}]'.format(item.get_dropout_rate()))
                if item.get_dropout_rate() > individual_max_dropout:
                    individual_max_dropout = item.get_dropout_rate()
            elif item.type == 2:
                block.append('pool[{:.2f}]'.format(item.get_dropout_rate()))
            else:
                block.append('full[{:.2f}]'.format(item.get_dropout_rate()))
        block.append(individual.complexity)
        block.append(individual_max_dropout)
        block.append(individual.mean)
        the_new_block.append(block)
        if max_dropout < individual_max_dropout:
            max_dropout = individual_max_dropout
            max_index = [index_individual]
        elif max_dropout == individual_max_dropout:
            max_index.append(index_individual)
        logger.info(block)
    if len(max_index) > 1:
        if len(the_new_population[max_index[0]].individual) > len(the_new_population[max_index[1]].individual):
            logger.info('==>, %s, ==> %s', max_index[0], the_new_block[max_index[0]])
            the_best_individual.append(the_new_population[max_index[0]])
        elif len(the_new_population[max_index[0]].individual) < len(the_new_population[max_index[1]].individual):
            logger.info('==>, %s, ==> %s', max_index[1], the_new_block[max_index[1]])
            the_best_individual.append(the_new_population[max_index[1]])
        else:
            logger.info('==>, %s, ==> %s', max_index[0], the_new_block[max_index[0]])
            the_best_individual.append(the_new_population[max_index[0]])
    else:
        logger.info('==>, %s, ==> %s', max_index[0], the_new_block[max_index[0]])
        the_best_individual.append(the_new_population[max_index[0]])
    logger.info('-'*16)
    if not os.path.exists(config.ckpt_save_dir):
        os.mkdir(config.ckpt_save_dir)

    flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR

    fd = os.open(os.path.join(config.ckpt_save_dir, 'the_best_individuals.dat'), flags, modes)
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(the_best_individual, f)
