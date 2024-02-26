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
import pickle
import stat
import numpy as np
from mindspore import log as logger


def extract_rectangles(dir_path):
    # Extract the zip file
    logger.info('Extracting the dataset')
    import zipfile
    fh = open(os.path.join(dir_path, 'rectangles_images.zip'), 'rb')
    z = zipfile.ZipFile(fh)

    dir_path = os.path.join(dir_path, 'rectanglesImages')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for name in z.namelist():
        s = name.split('/')
        flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR
        fd = os.open(os.path.join(dir_path, s[len(s) - 1]), flags, modes)
        outfile = os.fdopen(fd, "wb")
        outfile.write(z.read(name))
        outfile.close()
    fh.close()

    train_file_path = os.path.join(dir_path, 'rectangles_im_train.amat')
    valid_file_path = os.path.join(dir_path, 'rectangles_im_valid.amat')

    # Split data in valid file and train file
    fp = open(train_file_path)

    # Add the lines of the file into a list
    line_list = []
    for line in fp:
        line_list.append(line)
    fp.close()

    # Create valid file and train file
    flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR

    fd = os.open(valid_file_path, flags, modes)
    valid_file = os.fdopen(fd, "w")

    fd = os.open(train_file_path, flags, modes)
    train_file = os.fdopen(fd, "w")

    # Write lines into valid file and train file
    for i, line in enumerate(line_list):
        if (i + 1) > 10000:
            valid_file.write(line)
        else:
            train_file.write(line)

    valid_file.close()
    train_file.close()


def divide_dataset(data_np, data_length, k):
    data = data_np.reshape([data_length, -1])
    data_image = data[:, :-1].reshape([data_length, k, k])
    data_label = data[:, -1]
    data_label = np.array(data_label, dtype=np.int)
    return data_image, data_label


def load_rectangles_im(dir_path):
    data_save_path = os.path.join(dir_path, 'rectanglesImages_py.dat')
    if os.path.exists(data_save_path):
        with open(data_save_path, 'rb') as f:
            [(train_image, train_label), (valid_image, valid_label), (test_image, test_label)] = pickle.load(f)
        return (train_image, train_label), (valid_image, valid_label), (test_image, test_label)
    train_file_path = os.path.join(dir_path, 'rectangles_im_train.amat')
    valid_file_path = os.path.join(dir_path, 'rectangles_im_valid.amat')
    test_file_path = os.path.join(dir_path, 'rectangles_im_test.amat')
    assert os.path.exists(train_file_path)
    assert os.path.exists(valid_file_path)
    assert os.path.exists(test_file_path)

    # read data set
    with open(os.path.expanduser(train_file_path)) as f:
        string = f.read()
        train_data = np.array([float(i) for i in string.split()])
    train_image, train_label = divide_dataset(train_data, 10000, 28)
    with open(os.path.expanduser(valid_file_path)) as f:
        string = f.read()
        valid_data = np.array([float(i) for i in string.split()])
    valid_image, valid_label = divide_dataset(valid_data, 2000, 28)
    with open(os.path.expanduser(test_file_path)) as f:
        string = f.read()
        test_data = np.array([float(i) for i in string.split()])
    test_image, test_label = divide_dataset(test_data, 50000, 28)

    # save data file
    flags, modes = os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR

    fd = os.open(data_save_path, flags, modes)

    with os.fdopen(fd, 'wb') as f:
        pickle.dump([(train_image, train_label), (valid_image, valid_label), (test_image, test_label)], f)

    return (train_image, train_label), (valid_image, valid_label), (test_image, test_label)
