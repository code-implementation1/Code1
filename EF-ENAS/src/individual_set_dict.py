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
"""individual dic"""

import os
import pickle
import stat
from model_utils.config import config


class IndividualHistory:
    def __init__(self, path=None):
        self.individual_dict = {}
        if not path:
            if not os.path.exists(config.history):
                os.mkdir(config.history)
            if not os.path.exists(config.individual_history_path):
                os.mkdir(config.individual_history_path)
            self.path = os.path.join(config.individual_history_path, 'individual_dict.dat')
        else:
            self.path = os.path.abspath(path)

    def load_individual_history(self):
        with open(self.path, 'rb') as f:
            self.individual_dict = pickle.load(f)

    def save_individual_history(self):
        flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR

        fd = os.open(self.path, flags, modes)
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(self.individual_dict, f)

    def check_id(self, individual_id):
        if individual_id in self.individual_dict:
            if len(self.individual_dict[individual_id]) < 10:
                return 'ok'
            return 'no'
        return False

    def get_id(self, individual_id):
        message = self.individual_dict[individual_id]
        return message

    def add_id(self, individual_id, value):
        if individual_id in self.individual_dict:
            self.individual_dict[individual_id].append(value)
        else:
            self.individual_dict[individual_id] = [value]

    def extend_id_list(self, individual_dict):
        for key in individual_dict:
            if key in self.individual_dict:
                self.individual_dict[key].extend(individual_dict[key])
            else:
                self.individual_dict[key] = individual_dict[key]
