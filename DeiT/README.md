# Contents

* [Contents](#contents)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Environment Requirements](#environment-requirements)
* [Quick Start](#quick-start)
    * [Prepare the model](#prepare-the-model)
    * [Run the scripts](#run-the-scripts)
* [Script Description](#script-description)
    * [Script and Sample Code](#script-and-sample-code)
        * [Directory structure](#directory-structure)
        * [Script Parameters](#script-parameters)
    * [Training Process](#training-process)
        * [Training on GPU](#training-on-gpu)
            * [Training on multiple GPUs](#training-on-multiple-gpus)
            * [Training on single GPU](#training-on-single-gpu)
            * [Arguments description](#arguments-description)
        * [Training with CPU](#training-with-cpu)
        * [Transfer training](#transfer-training)
    * [Evaluation](#evaluation)
        * [Evaluation process](#evaluation-process)
            * [Evaluation with checkpoint](#evaluation-with-checkpoint)
            * [Evaluation with ONNX](#evaluation-with-onnx)
        * [Evaluation results](#evaluation-results)
    * [Inference](#inference)
        * [Inference with checkpoint](#inference-with-checkpoint)
        * [Inference with ONNX](#inference-with-onnx)
        * [Inference results](#inference-results)
    * [Export](#export)
        * [Export process](#export-process)
        * [Export results](#export-results)
* [Model Description](#model-description)
    * [Performance](#performance)
        * [Training Performance](#training-performance)
* [Description of Random Situation](#description-of-random-situation)
* [ModelZoo Homepage](#modelzoo-homepage)

# [DeiT Description](#contents)

DeiT: Data-efficient Image Transformers

This repository contains Mindspore evaluation code, training code and pretrained models for the DeiT (Data-Efficient Image Transformers), ICML 2021

They obtain competitive tradeoffs in terms of speed / precision

[Paper](https://arxiv.org/abs/2012.12877): Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles and Hervé Jégou, Training data-efficient image transformers & distillation through attention.

# [Model Architecture](#contents)

The larger model, DeiT-B, has thesame architecture as the ViT-B.
The only parameters that vary across models are the embedding dimension and
the number of heads, and we keep the dimension per head constant (equal to 64).
Smaller models have a lower parameter count, and a faster throughput.

Model contains PatchEmbed layer, after that 12 Block layers.
Each block contain self-attention layer, DropPath and MLP layers.
After 12 applications of this block, it applied LayerNorm and Dense
layer get a feature vector, which is passed to a softmax classifier.

There are following configurations:

* deit_base_patch16_224
* deit_tiny_patch16_224
* deit_small_patch16_224
* deit_base_patch16_384
* deit_base_distilled_patch16_224
* deit_tiny_distilled_patch16_224
* deit_small_distilled_patch16_224
* deit_base_distilled_patch16_384

The difference between all of them in the following:

| Model                           | num heads | embed dims | input size | distilled token |
|---------------------------------|-----------|------------|------------|-----------------|
| deit_base_patch16_224           | 12        | 768        | 224x224    | NO              |
| deit_tiny_patch16_224           | 3         | 192        | 224x224    | NO              |
| deit_small_patch16_224          | 6         | 384        | 224x224    | NO              |
| deit_base_patch16_384           | 12        | 768        | 384x384    | NO              |
| deit_base_distilled_patch16_224 | 12        | 768        | 224x224    | YES             |
| deit_tiny_distilled_patch16_224 | 3         | 192        | 224x224    | YES             |
| deit_small_distilled_patch16_224| 6         | 384        | 224x224    | YES             |
| deit_base_distilled_patch16_384 | 12        | 768        | 384x384    | YES             |

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [ImageNet2012](http://www.image-net.org/)

* Dataset size：146.6G
    * Train：139.3G，1281167 images
    * Val：6.3G，50000 images
    * Annotations：each image is in label folder
* Data format：images sorted by label folders
    * Note：Data will be processed in imagenet.py

# [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset ImageNet dataset.
* We use ImageNet2012 as training dataset in this example by default, and you
  can also use your own datasets.

For ImageNet-like dataset the directory structure is as follows:

```shell
 .
 └─imagenet
   ├─train
     ├─class1
       ├─image1.jpeg
       ├─image2.jpeg
       └─...
     ├─...
     └─class1000
   ├─val
     ├─class1
     ├─...
     └─class1000
   └─test
```

# [Quick Start](#contents)

## Prepare the model

1. Chose the model by changing the `model` in `configs/deit_xxx_patch16_yyy.yaml`, where `xxx` -- model name, `yyy` -- image size.
   Allowed options are:
   `deit_base_patch16_224`
   `deit_tiny_patch16_224`
   `deit_small_patch16_224`
   `deit_base_patch16_384`
   `deit_base_distilled_patch16_224`
   `deit_tiny_distilled_patch16_224`
   `deit_small_distilled_patch16_224`
   `deit_base_distilled_patch16_384`
2. Change the dataset config in the corresponding config. `configs/deit_xxx_patch16_yyy.yaml`.
   Especially, set the correct path to data.
3. Change the hardware setup.
4. Change the artifacts setup to set the correct folders to save checkpoints and mindinsight logs.

Note, that you also can pass the config options as CLI arguments, and they are
preferred over config in YAML.

## Run the scripts

After installing MindSpore via the official website,
you can start training and evaluation as follows.

```shell
# distributed training on GPU
bash run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# standalone training on GPU
bash run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# run eval on GPU
bash run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

### Directory structure

```shell
DeiT
├── scripts
│   ├── run_distribute_train_gpu.sh                          # shell script for distributed training on GPU
│   ├── run_eval_gpu.sh                                      # shell script for evaluation on GPU
│   ├── run_eval_onnx.sh                                     # shell script for evaluation with ONNX model
│   ├── run_infer_gpu.sh                                     # shell script for inference on GPU
│   ├── run_infer_onnx.sh                                    # shell script for inference with ONNX model
│   └── run_standalone_train_gpu.sh                          # shell script for training on GPU
├── src
│  ├── configs
│  │  ├── deit_base_patch16_224.yaml                         # example of configuration for deit_base_patch16_224
│  │  └── deit_base_patch16_224_distilled.yaml               # example of configuration for deit_base_distilled_patch16_224
│  │  
│  ├── data
│  │  ├── augment
│  │  │  ├── __init__.py
│  │  │  ├── auto_augment.py                                 # augmentation set builder
│  │  │  ├── mixup.py                                        # MixUp augmentation
│  │  │  ├── transforms.py                                   # some transforms for augmentations
│  │  │  └── random_erasing.py                               # Random Erasing augmentation
│  │  ├── data_utils
│  │  │  ├── __init__.py
│  │  │  └── moxing_adapter.py                               # DS synchronization for distributed training
│  │  ├── __init__.py
│  │  ├── constants.py                                       # Imagenet data constants
│  │  └── imagenet.py                                        # wrapper for reading ImageNet dataset
│  ├── model
│  │  ├── layers
│  │  │  ├── __init__.py
│  │  │  ├── adaptive_avgmax_pool.py                         # adaptive_avgmax_pool layer
│  │  │  ├── attention.py                                    # attention layer
│  │  │  ├── block.py                                        # block layer
│  │  │  ├── classifier.py                                   # custom_identity layer
│  │  │  ├── conv2d_same.py                                  # conv2d_same layer
│  │  │  ├── conv_bn_same.py                                 # conv_bn_same layer
│  │  │  ├── conv_bn_act.py                                  # conv_bn_act layer
│  │  │  ├── create_norm_act.py                              # norm_act layer
│  │  │  ├── custom_identity.py                              # custom_identity layer
│  │  │  ├── distilled_vision_transformer.py                 # distilled_vision_transformer model
│  │  │  ├── drop_path.py                                    # drop_path layer
│  │  │  ├── mlp.py                                          # mlp layer
│  │  │  ├── norm_act.py                                     # norm_act layer
│  │  │  ├── padding.py                                      # padding layer
│  │  │  ├── patch_embed.py                                  # patch_embed layer
│  │  │  ├── regnet.py                                       # regnet layer
│  │  │  ├── se.py                                           # se layer
│  │  │  ├── vision_transformer.py                           # vision transformer
│  │  │  └── weights_init.py                                 # tools for init weights
│  │  └── factory.py                                         # define models
│  │
│  ├── tools
│  │  ├── __init__.py
│  │  ├── callback.py                                        # callback functions (implementation)
│  │  ├── cell.py                                            # tune model layers/parameters
│  │  ├── common.py                                          # function for get whole set of callbacks
│  │  ├── criterion.py                                       # model training objective function (implementation)
│  │  ├── get_misc.py                                        # initialize optimizers and other arguments for training process
│  │  ├── optimizer.py                                       # model optimizer function (implementation)
│  │  └── schedulers.py                                      # training (LR) scheduling function (implementation)
│  ├── trainer
│  │  ├── ema.py                                             # EMA implementation
│  │  ├── train_one_step_with_ema.py                         # utils for training with EMA
│  │  └── train_one_step_with_scale_and_clip_global_norm.py  # utils for training with gradient clipping
│  └── args.py                                               # YAML and CLI configuration parser
│
├── convert_pt_ro_ms.py                                      # converting pytorch checkpoints to mindspore
├── eval.py                                                  # evaluation script
├── eval_onnx.py                                             # evaluation script for ONNX model
├── export.py                                                # export checkpoint files into MINDIR, ONNX and AIR formats
├── infer.py                                                 # inference script
├── infer_onnx.py                                            # inference script for ONNX model
├── README.md                                                # DeiT descriptions
├── requirements.txt                                         # python requirements
└── train.py                                                 # training script
```

### [Script Parameters](#contents)

```yaml
# ===== Model ===== #
model: deit_base_patch16_224

finetune: ''
exclude_epoch_state: false

# ===== Dataset ===== #
data_path: /imagenet/

train_dir: train
val_dir: validation_preprocess
num_classes: 1000
input_size: 224
no_dataset_sink_mode: false


# ===== Base training config ===== #
amp_level: O2
lr: 0.0005
min_lr: 1.0e-05
batch_size: 64
epochs: 300
model_ema: false


# ===== Network training config ===== #
clip_grad: null
clip_grad_norm: 5.0
bce_loss: false

momentum: 0.9
num_workers: 10
opt: adamw
opt_eps: 1.0e-08
unscale_lr: false
use_clip_grad_norm: false
warmup_epochs: 5
warmup_lr: 1.0e-06
weight_decay: 0.05
patience_epochs: 10
start_epoch: 0
sched: cosine_lr
decay_epochs: 30
decay_rate: 0.1
is_dynamic_loss_scale: 1
loss_scale: 1024
cooldown_epochs: 10


# ===== Hardware setup ===== #
device_id: 0
device_num: 1
device_target: GPU


# ===== Augments ===== #
smoothing: 0.1
eval_crop_ratio: 0.875
aa: rand-m9-mstd0.5-inc1
beta:
- 0.9
- 0.999
recount: 1
remode: pixel
reprob: 0.25
resplit: false
color_jitter: 0.3
cutmix: 1.0
cutmix_minmax: null
mixup: 0.8
mixup_mode: batch
mixup_off_epoch: 0
mixup_prob: 1.0
mixup_switch_prob: 0.5
interpolation: bicubic




# ===== EMA ===== #
model_ema_decay: 0.99996

# ===== Distilation ===== #

distillation_alpha: 0.5
distillation_tau: 1.0
distillation_type: none
teacher_model: regnety_160
teacher_path: ''


# ===== Artifacts setup ===== #
keep_checkpoint_max: 10
output_dir: ./
ckpt_keep_num: 10
ckpt_save_every_sec: 3600
ckpt_save_every_step: 0
collect_input_data: false
summary_loss_collect_freq: 1
print_loss_every: 20

```

## [Training Process](#contents)

In the examples below the only required argument is YAML config file.

### Training on GPU

#### Training on multiple GPUs

Usage

```shell
# distributed training on GPU
run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments
bash run_distribute_train.sh ../src/configs/deit_base_patch16_224.yaml --num_devices 4 --device_ids 0,1,2,3

# With extra arguments
bash run_distribute_train.sh ../src/configs/deit_base_patch16_224.yaml --num_devices 4 --device_ids 0,1,2,3 --extra --amp_level O0 --batch_size 64 --start_epoch 0 --num_parallel_workers 8
```

#### Training on single GPU

Usage

```shell
# standalone training on GPU
run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments:
bash run_standalone_train.sh ../src/configs/deit_base_patch16_224.yaml --device 0
# With extra arguments:
bash run_standalone_train.sh ../src/configs/deit_base_patch16_224.yaml --device 0 --extra --amp_level O0 --batch_size 64 --start_epoch 0 --num_parallel_workers 8
```

Running the Python scripts directly is also allowed.

```shell
# show help with description of options
python train.py --help

# standalone training on GPU
python train.py --config path/to/config.yaml [OTHER OPTIONS]
```

#### Arguments description

`bash` scripts have the following arguments

* `CONFIG`: path to YAML file with configuration.
* `--num_devices`: the device number for distributed train.
* `--device_ids`: ids of devices to train.
* `--checkpoint`: path to checkpoint to continue training from.
* `--extra`: any other arguments of `train.py`.

By default, training process produces three folders (configured):

* Best checkpoints
* Current checkpoints
* Mindinsight logs

### Training with CPU

**It is recommended to run models on GPU.**

### Transfer training

You can train your own model based on pretrained classification
model. You can perform transfer training by following steps.

1. Convert your own dataset to ImageFolderDataset style. Otherwise, you have to add your own data preprocess code.
2. Change `deit_xxx_patch16_yyy.yaml` according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `pretrained` argument.
4. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation](#contents)

### Evaluation process

#### Evaluation with checkpoint

Usage

```shell
run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Examples

```shell

# Without extra args
bash run_eval_gpu.sh  ../src/configs/deit_base_patch16_224.yaml --checkpoint /data/models/deit_base_patch16_224.ckpt

# With extra args
bash run_eval_gpu.sh  ../src/configs/deit_base_patch16_224.yaml --checkpoint /data/models/deit_base_patch16_224.ckpt --extra --data_url /data/imagenet/ --val_dir validation_preprocess
```

Running the Python script directly is also allowed.

```shell
# run eval on GPU
python eval.py --config path/to/config.yaml [OTHER OPTIONS]
```

The Python script has the same arguments as the training script (`train.py`),
but it uses only validation subset of dataset to evaluate.
Also, `--pretrained` is expected.

#### Evaluation with ONNX

Usage

```shell
run_eval_onnx.sh DATA [--onnx_path ONNX_PATH]
```

* `DATA` is a test subset
* `--onnx_path` is path to ONNX model.

Example

```bash
bash run_eval_onnx.sh /data/imagenet/val --onnx_path /data/models/deit_base_patch16_224.onnx
```

Also, Python script may be used.

Usage

```shell
eval_onnx.py [-h] [--onnx_path ONNX_PATH] [--image_size IMAGE_SIZE (default: 224)]
             [-m {CPU,GPU} (default: GPU)] [--prefetch PREFETCH (DEFAULT: 16)]
             dataset
```

Example

```shell
python eval_onnx.py /data/imagenet/val --onnx_path deit_base_patch16_224.onnx
```

### Evaluation results

Results will be printed to console.

```shell
# checkpoint evaluation result
eval results: {'Loss': 1.4204, 'Top1-Acc': 0.731066, 'Top5-Acc': 0.90621}

# ONNX evaluation result
eval results: {'Top1-Acc': 0.731066, 'Top5-Acc': 0.90621}
```

## [Inference](#contents)

Inference may be performed with checkpoint or ONNX model.

### Inference with checkpoint

Usage

```shell
run_infer_gpu.sh DATA [--checkpoint CHECKPOINT] [--model ARCHITECTURE] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example for folder

```shell
bash run_infer_gpu.sh /data/images/cheetah/ --checkpoint /data/models/deit_base_patch16_224.ckpt --model deit_base_patch16_224
```

Example for single image

```shell
bash run_infer_gpu.sh /data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG --checkpoint /data/models/deit_base_patch16_224_trained.ckpt --model deit_base_patch16_224
```

### Inference with ONNX

Usage

```bash
run_infer_onnx.sh DATA [--onnx_path ONNX_PATH] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example

```bash
bash run_infer_onnx.sh /data/images/cheetah/ --onnx_path /data/models/deit_base_patch16_224.onnx
```

### Inference results

Predictions will be output in logs and saved in JSON file. Predictions format
is same for mindspore and ONNX model File content is dictionary where key is
image path and value is class number. It's supported predictions for folder of
images (png, jpeg file in folder root) and single image.

Results for single image in console

```shell
/data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG (class: 293)
```

Results for single image in JSON file

```json
{
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG": 293
}
```

Results for directory in console

```shell
/data/images/cheetah/ILSVRC2012_validation_preprocess_00033907.JPEG (class: 293)
/data/images/cheetah/ILSVRC2012_validation_preprocess_00033988.JPEG (class: 293)
/data/images/cheetah/ILSVRC2012_validation_preprocess_00013656.JPEG (class: 293)
/data/images/cheetah/ILSVRC2012_validation_preprocess_00038707.JPEG (class: 293)
```

Results for directory in JSON file

```json
{
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00033907.JPEG": 293,
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00033988.JPEG": 293,
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00013656.JPEG": 293,
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00038707.JPEG": 293
}
```

## [Export](#contents)

### Export process

Trained checkpoints may be exported to `MINDIR`, `AIR` (currently not checked) and `ONNX`.

NOTE: In the model uses layer `nn.GELU` with parameter `approximate=False`.
In current version of the mindspore it does not supporteed during exporting to `ONNX` .
So it will raise error when you try to run the command below:

```shell
python export.py --file-format FILE_FORMAT --checkpoint-path path/to/checkpoint.ckpt --model ARCHITECTURE_NAME
```

We can use `nn.GELU` with parameter `approximate=True`, but in this case it
have a little difference between original implementation and this.
We added flag `--fix-gelu`, which allow us to import model, but you need to
understand that it will export model with differences from original model.

Example

```shell

# Export to MINDIR
python export.py --config src/configs/deit_base_patch16_224.yaml --file-format MINDIR --checkpoint-path /data/models/deit_base_patch16_224.ckpt --model deit_base_patch16_224 --fix-gelu

# Export to ONNX
python export.py --config src/configs/deit_base_patch16_224.yaml --file-format ONNX --checkpoint-path /data/models/deit_base_patch16_224.ckpt --model deit_base_patch16_224 --fix-gelu
```

### Export results

Exported models saved in the current directory with name the same as architecture.

# [Model Description](#contents)

## Performance

### Training Performance

| Parameters                 | GPU                                    |
|----------------------------|----------------------------------------|
| Model Version              | deit_base_patch16_224                  |
| Resource                   | 4xGPU (NVIDIA GeForce RTX 3090)        |
| Uploaded Date              | 12/26/2023 (month/day/year)            |
| MindSpore Version          | 1.9.0                                  |
| Dataset                    | ImageNet                               |
| Training Parameters        | src/configs/deit_base_patch16_224.yaml |
| Optimizer                  | AdamW                                  |
| Loss Function              | SoftmaxCrossEntropy                    |
| Outputs                    | logits                                 |
| Accuracy                   | ACC1 [~0.71]                           |
| Total time                 |                                        |
| Params (M)                 |                                        |
| Checkpoint for Fine tuning |                                        |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).