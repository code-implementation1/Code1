# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [Contents](#contents)
- [EF-ENAS Description](#ef-enas-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Network Architecture Search Process](#network-architecture-search-process)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [export model](#export-model)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# EF-ENAS Description

EF-ENAS is an evolutionary neural architecture search algorithm based on evaluation correction and functional units proposed in 2022 to search for network structures with excellent performance on a specific dataset.

[Evolutionary neural architecture search based on evaluation correction and functional units](https://doi.org/10.1016/j.knosys.2022.109206)

# Model Architecture

The searched network structure consists of various network blocks, including 3x3 convolution, 2x2 maximum pooling, Dropout, four different activation functions (crelu, relu, elu, tanh) and two network weight initialization methods (xavier uniform, he uniform).

# Dataset

Dataset used：[RectanglesImages](<https://drive.google.com/file/d/1lmIE2zH2tAUBiivRwe-cQc5nFf3F1S0X/view?usp=sharing>)

- Dataset size: 389M, 62,000 28*23 grayscale images in 2 classes
    - Train: 12,000 images
    - Test: 50,000 images
- Data format：binary files

Download the dataset and place it in the directory shown below.

```shell
./rectanglesImages
```

# Environment Requirements

- Hardware (NPU)
    - Prepare hardware environment with NPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on NPU

```shell

# run network architecture search
bash scripts/run_evolution.sh [data_path] [ckpt_save_dir] [history] [output_path] [individual_history_path] [offspring_path] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_evolution.sh ./rectanglesImages/ ./the_best_model/ ./history/ ./history/cache/ ./history/individual_history/ ./history/offspring/ 0.0005 128 10

# save the best network architecture from the population
bash scripts/run_save_model_arch.sh

# train the network architecture searched
bash scripts/run_train.sh [data_path] [architecture_dir] [ckpt_save_dir] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_train.sh ./rectanglesImages/ ./the_best_model/architecture/ the_best_model/ 0.0005 128 10

# Deep training of the searched network structure using distributed
bash scripts/run_train-dis.sh [RANK_TABLE_FILE]
# such as bash scripts/run_train-dis.sh hccl_8p.json

# evaluate the network architecture trained
bash scripts/run_eval.sh [data_path] [architecture_dir] [ckpt_save_dir] [batch_size]
# bash scripts/run_eval.sh rectanglesImages/ the_best_model/architecture/ the_best_model/ 128
```

For distributed training, the HCCL configuration file in JSON format needs to be created in advance, please follow the instructions in the link below:
https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.

# Script Description

## Script and Sample Code

```bash
├── model_zoo
    ├── README.md                           // descriptions about all the models
    ├── EF_ENAS
        ├── README_CN.md                    // descriptions about EF-ENAS[Chinese]
        ├── README.md                       // descriptions about EF-ENAS[English]
        ├── src
        │   ├── check_networks.py           // check network architecture
        │   ├── evaluate.py                 // evaluate individual
        │   ├── Individual.py               // build individual
        │   ├── individual_set_dict.py      // record the network architecture on evolution
        │   ├── Layers.py                   // build network layer
        │   └── Population.py               // build population
        ├── data
        │   ├── get_data.py                 // creating dataset
        │   └── rectanglesImagesDataSet.py  // creating dataset
        ├── rectanglesImages
        │   ├── ReadMe.md                   // descriptions about dataset and download
        │   └── rectanglesImages_py.dat     // dataset
        ├── scripts
        │   ├── run_evolution.sh            // shell script for starting evolution
        │   ├── run_save_model_arch.sh      // shell script for saving network architecture searched
        │   ├── run_train.sh                // shell script for training network architecture searched
        │   ├── run_train-dis.sh            // shell script for training network architecture searched using distributed
        │   └── run_eval.sh                 // shell script for evaluating network architecture searched
        ├── model_utils
        │   ├── config.py                   // parameters config
        │   ├── device_adapter.py           // device adapter
        │   └── local_adapter.py            // local adapter
        ├── the_best_model
        │   ├── architecture
        │   │   └── the_best_individuals.dat // the best architecture searched
        │   └── the_best_individuals.ckpt   // the weights of the best architecture trained
        ├── evolution.py                    // search network architecture
        ├── save_model_arch_for_training.py // save the network architecture searched
        ├── train.py                        // train network architecture searched
        ├── export.py                       // exoirt model AIR/MNIDIR
        ├── evolution.yaml                  // the configure file
        └── eval.py                         // evaluate network architecture searched
```

## Network Architecture Search Process

```bash
bash scripts/run_evolution.sh [data_path] [ckpt_save_dir] [history] [output_path] [individual_history_path] [offspring_path] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_evolution.sh ./rectanglesImages/ ./the_best_model/ ./history/ ./history/cache/ ./history/individual_history/ ./history/offspring/ 0.0005 128 10
bash scripts/run_save_model_arch.sh
```

The above python commands complete the search and save of the network structure. The search process of the network structure is saved in the `history` folder, and the best network structure is saved in the `the_best_model` folder.

## Training Process

### Training

- on the Ascend

```bash
bash scripts/run_train.sh [data_path] [architecture_dir] [ckpt_save_dir] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_train.sh ./rectanglesImages/ ./the_best_model/architecture/ the_best_model/ 0.0005 128 10
```

The above python command completes the training of the searched network structure. And after the training, the model parameters are saved in `. /the_best_model/EF-ENAS.ckpt`.

- on the distributed training

```shell
base scripts/run_train-dis.sh ./hccl_8p.json
```

## Evaluation Process

### Evaluation

```bash
bash scripts/run_eval.sh [data_path] [architecture_dir] [ckpt_save_dir] [batch_size]
# bash scripts/run_eval.sh rectanglesImages/ the_best_model/architecture/ the_best_model/ 128
```

The above python command completes the performance evaluation of the trained network structure, and outputs the evaluation results on the command line.

## export model

```shell
python export.py --ckpt_file=[CKPT_PATH] --file_format=[MINDIR, AIR]
```

# Model Description

## Performance

### Training Performance

Training the architecture searched on RectanglesImages

| Parameters          | Ascend 910                           |
|---------------------|--------------------------------------|
| Model Version       | EF-ENAS                              |
| Resource            | Ascend 910; system Ubuntu18.04       |
| Uploaded Date       | 2023-10-20                           |
| MindSpore Version   | 1.8.0                                |
| Dataset             | RectanglesImages                     |
| Training Parameters | epoch=200, batch_size=128, lr=0.0005 |
| Optimizer           | Adam                                 |
| Loss Function       | Softmax Cross Entropy                |
| outputs             | probability                          |
| Loss                | 0.047                                |
| Speed               | 21ms/step                            |
| Total time          | 1pc:2day7hour17minutes               |

# [ModelZoo Homepage]

Please check the official [homepage](https://gitee.com/mindspore/models).  
