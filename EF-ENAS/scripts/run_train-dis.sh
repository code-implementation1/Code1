#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

# 如果参数1指向的文件不存在
if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

# 设置超级用户root的内存和线程限制为无线
ulimit -u unlimited

# 多NPU设置
export DEVICE_NUM=8
export RANK_SIZE=8
# 获取参数1指向文件的绝对路径
PATH1=$(realpath $1)
export RANK_TABLE_FILE=$PATH1
echo "RANK_TABLE_FILE=${PATH1}"

# 服务器ID
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

# cpu核心数
cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
# 平均每个NPU设备能分到几个cpu核心
avg=`expr $cpus \/ $DEVICE_NUM`
# 每次从第一个到最后一个出的增量,减掉第一个
gap=`expr $avg \- 1`

# 对每个NPU进行设置
for((i=0; i<${DEVICE_NUM}; i++))
do
    # 开始核结束的cpu编号
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    # 设备号`
    export DEVICE_ID=$i
    # rank ID
    export RANK_ID=$((rank_start + i))
    # 确保 ./train_parallel_i 文件夹为空
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    # 复制 src, model_utils, *.yaml, train.py 路径和文件到 train_parallel_i 路径下
    cp -r ./src ./train_parallel$i
    cp -r ./data ./train_parallel$i
    cp -r ./rectanglesImages ./train_parallel$i
    cp -r ./the_best_model ./train_parallel$i
    cp -r ./model_utils ./train_parallel$i
    cp -r ./*.yaml ./train_parallel$i
    cp ./*.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    # 进入改文件夹, 如果该文件夹创建失败则退出脚本
    cd ./train_parallel$i ||exit
    env > env.log
    # 将某个进程任务指定到某个cpu核心上进行
    # -c 通过列表现实方式设置cpu (逗号相隔)
    # 指定数据集目录, 数据集, 是否进行分布式
    taskset -c $cmdopt python train.py --is_distributed True > log 2>&1 &
    cd ..
done
