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

#python3 eval.py     --data_path "./rectanglesImages" \
#                    --architecture_dir "./the_best_model/architecture" \
#                    --ckpt_save_dir "./the_best_model/" \
#                    --batch_size 128
if [ $# != 4 ]
then
  echo "input parameters Err."
  echo "[data_path] [architecture_dir] [ckpt_save_dir] [batch_size]"
fi
python3 eval.py --data_path $1 --architecture_dir $2 --ckpt_save_dir $3 --batch_size $4
