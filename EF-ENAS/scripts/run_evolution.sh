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

#python3 evolution.py  --data_path "./rectanglesImages" \
#                      --ckpt_save_dir "./the_best_model/" \
#                      --history "./history" \
#                      --output_path "./history/cache" \
#                      --individual_history_path "./history/individual_history" \
#                      --offspring_path "./history/offspring" \
#                      --learning_rate 0.0005 \
#                      --batch_size 128 \
#                      --epoch_size 10

if [ $# != 9 ]
then
  echo "input parameters Err."
  echo "[data_path] [ckpt_save_dir] [history] [output_path] [individual_history_path] [offspring_path] [learning_rate] [batch_size] [epoch_size]"
fi
python3 evolution.py --data_path $1 --ckpt_save_dir $2 --history $3 --output_path $4 --individual_history_path $5 --offspring_path $6 --learning_rate $7 --batch_size $8 --epoch_size $9
