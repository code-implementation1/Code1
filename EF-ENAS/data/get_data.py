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
"""generate data set"""

import os
import mindspore.dataset as ds
import numpy as np
from data.rectangles_images_dataset import load_rectangles_im


def _load_rectangles_im(path=None, is_valid=False):
    # read data
    if not path:
        path = os.path.abspath(r'./rectanglesImages')
    (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = load_rectangles_im(path)
    # deal with valid dataset
    if is_valid:
        return (train_images, train_labels), (valid_images, valid_labels)
    train_images = np.concatenate([train_images, valid_images], 0)
    train_labels = np.concatenate([train_labels, valid_labels], 0)

    return (train_images, train_labels), (test_images, test_labels)


def _normalized_image(images_set, data_shape):
    images_set = images_set.astype(np.float32)
    images_set_len = len(images_set)
    for image_index in range(images_set_len):
        mean = np.mean(images_set[image_index])
        var = np.sqrt(np.var(images_set[image_index]))
        if var > 1.0/data_shape[0]:
            tmp_var = var
        else:
            tmp_var = 1.0/data_shape[0]
        images_set[image_index] = (images_set[image_index] - mean)/tmp_var

    return images_set


class RIDataset:
    def __init__(self, data_path=r'./rectanglesImages', mode='train', is_valid=True):
        if mode == 'train':
            (data, label), _ = _load_rectangles_im(data_path, is_valid=is_valid)
        elif mode == 'test':
            _, (data, label) = _load_rectangles_im(data_path, is_valid=is_valid)
        else:
            raise ValueError('split mode Err.')
        data = _normalized_image(data, data_shape=[28, 28])
        self.data = np.reshape(data, [-1, 1, 28, 28])
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def get_ri(data_path, mode, is_valid, batch_size):
    dataset_generator = RIDataset(data_path=data_path, mode=mode, is_valid=is_valid)
    dataset = ds.GeneratorDataset(dataset_generator, ['data', 'label'], shuffle=True)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset


def get_ri_for_search(data_path, batch_size):
    train_set = get_ri(data_path=data_path, mode='train', is_valid=True, batch_size=batch_size)
    valid_set = get_ri(data_path=data_path, mode='test', is_valid=True, batch_size=batch_size)

    return train_set, valid_set


def get_ri_for_evaluate(data_path, batch_size):
    train_set = get_ri(data_path=data_path, mode='train', is_valid=False, batch_size=batch_size)
    valid_set = get_ri(data_path=data_path, mode='test', is_valid=False, batch_size=batch_size)

    return train_set, valid_set
