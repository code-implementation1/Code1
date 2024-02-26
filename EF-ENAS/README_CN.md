# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [EF-ENAS 描述](#ef-enas-描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [网络结构搜索过程](#网络结构搜索过程)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [模型导出](#模型导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# EF-ENAS 描述

EF-ENAS是2022年提出的一种基于评估矫正和功能单元的进化神经结构搜索算法，他能够在特定数据集上搜索出性能优异的网络结构。

[Evolutionary neural architecture search based on evaluation correction and functional units](https://doi.org/10.1016/j.knosys.2022.109206)

# 模型架构

搜索出的网络结构由多种网络块组成，包括3x3卷积，2x2最大池化，Dropout，四种不同的激活函数 (crelu, relu, elu, tanh) 和两种网络权重初始化方式 (xavier uniform, he uniform)

# 数据集

使用的数据集：[RectanglesImages](<https://drive.google.com/file/d/1lmIE2zH2tAUBiivRwe-cQc5nFf3F1S0X/view?usp=sharing>)

- 数据集大小：389M，共2个类，62万张28*28灰度图像
    - 训练集：共1.2万张图像
    - 测试集：共5万张图像
- 数据格式：二进制文件

下载数据集后放于如下所示目录

```shell
./rectanglesImages
```

# 环境要求

- 硬件（NPU）
    - 使用NPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)

# 快速入门

```shell  
# 运行神经网络结构搜索
bash scripts/run_evolution.sh [data_path] [ckpt_save_dir] [history] [output_path] [individual_history_path] [offspring_path] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_evolution.sh ./rectanglesImages/ ./the_best_model/ ./history/ ./history/cache/ ./history/individual_history/ ./history/offspring/ 0.0005 128 10

# 从搜索到的种群中得到最优的网络结构并保存
bash scripts/run_save_model_arch.sh

# 对搜索到的网络结构进行深训练
bash scripts/run_train.sh [data_path] [architecture_dir] [ckpt_save_dir] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_train.sh ./rectanglesImages/ ./the_best_model/architecture/ the_best_model/ 0.0005 128 10

# 使用分布式对搜索到的网络结构进行深训练
bash scripts/run_train-dis.sh [RANK_TABLE_FILE]
# such as bash scripts/run_train-dis.sh hccl_8p.json

# 评估训练后的模型精度
bash scripts/run_eval.sh [data_path] [architecture_dir] [ckpt_save_dir] [batch_size]
# bash scripts/run_eval.sh rectanglesImages/ the_best_model/architecture/ the_best_model/ 128
```

对于分布式训练, 需要提前创建JSON格式得HCCL配置文件, 请遵循以下链接中得说明:
https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── EF_ENAS
        ├── README_CN.md                    // 相关代码说明[中文]
        ├── README.md                       // 相关代码说明[英文]
        ├── src
        │   ├── check_networks.py           // 实现网络结构检查功能
        │   ├── evaluate.py                 // 实现网络结构个体的验证功能
        │   ├── Individual.py               // 实现网络结构构建的功能
        │   ├── individual_set_dict.py      // 实现种群个体产生历史记录的功能
        │   ├── Layers.py                   // 实现网络层构建的功能
        │   └── Population.py               // 实现网络结构种群构建的功能
        ├── data
        │   ├── get_data.py                 // 数据集构造
        │   └── rectanglesImagesDataSet.py  // 数据集构造
        ├── rectanglesImages
        │   ├── ReadMe.md                   // 数据集下载和使用说明文件
        │   └── rectanglesImages_py.dat     // 数据集存储文件
        ├── scripts
        │   ├── run_evolution.sh            // 进化网络结构搜索算法启动脚本
        │   ├── run_save_model_arch.sh      // 搜索到的网络结构保存脚本
        │   ├── run_train.sh                // 搜索到的网络结构训练脚本
        │   ├── run_train-dis.sh            // 搜索到的网络结构训练脚本，使用分布式训练
        │   └── run_eval.sh                 // 搜索到的网络结构验证脚本
        ├── model_utils
        │   ├── config.py                   // 参数配置
        │   ├── device_adapter.py           // device adapter
        │   └── local_adapter.py            // local adapter
        ├── the_best_model
        │   ├── architecture
        │   │   └── the_best_individuals.dat // 搜索到得最好得网络结构
        │   └── the_best_individuals.ckpt   // 训练好的网络结构得权重
        ├── eval.py                         // 执行网络结构验证
        ├── evolution.py                    // 执行进化网络结构搜索
        ├── evolution.yaml                  // 参数配置文件
        ├── save_model_arch_for_training.py // 保存搜索到的网络结构
        ├── export.py                       // 将模型导出到AIR/MNIDIR
        └── train.py                        // 执行网络结构训练
```

## 网络结构搜索过程

```bash
bash scripts/run_evolution.sh [data_path] [ckpt_save_dir] [history] [output_path] [individual_history_path] [offspring_path] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_evolution.sh ./rectanglesImages/ ./the_best_model/ ./history/ ./history/cache/ ./history/individual_history/ ./history/offspring/ 0.0005 128 10
bash scripts/run_save_model_arch.sh
```

上述python命令完成对网络结构的搜索和保存。其中网络结构的搜索过程保存在`history`文件夹下，搜索到的最优网络结构保存在`the_best_model`文件夹下。

## 训练过程

### 训练

- 在Ascend处理器环境运行

```bash
bash scripts/run_train.sh [data_path] [architecture_dir] [ckpt_save_dir] [learning_rate] [batch_size] [epoch_size]
#bash scripts/run_train.sh ./rectanglesImages/ ./the_best_model/architecture/ the_best_model/ 0.0005 128 10
```

上述python命令完成对搜索到的网络结构的训练，训练结束后，模型参数保存在`./the_best_model/the_best_individuals.ckpt`

- 分布式训练

```shell
base scripts/run_train-dis.sh ./hccl_8p.json
```

## 评估过程

### 评估

```bash
bash scripts/run_eval.sh [data_path] [architecture_dir] [ckpt_save_dir] [batch_size]
# bash scripts/run_eval.sh rectanglesImages/ the_best_model/architecture/ the_best_model/ 128
```

上述python命令完成对训练好的网络结构的性能评估，评估结果在命令行输出。

## 模型导出

```shell
python export.py --ckpt_file=[CKPT_PATH] --file_format=[MINDIR, AIR]
```

# 模型描述

## 性能

### 训练性能

rectanglesImages上训练搜索到的网络结构

| 参数          | Ascend 910                           |
| ----------- |--------------------------------------|
| 模型版本        | EF-ENAS                              |
| 资源          | Ascend 910; 系统 ubuntu18.04           |
| 上传日期        | 2023-02-22                           |
| MindSpore版本 | 1.8.0                                |
| 数据集         | RectanglesImages                     |
| 训练参数        | epoch=200, batch_size=128, lr=0.0005 |
| 优化器         | Adam                                 |
| 损失函数        | Softmax交叉熵                           |
| 输出          | 概率                                   |
| 损失          | 0.047                                |
| 速度          | 21毫秒/步                               |
| 总时长         | 1pc:2day7hour17minutes               |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
