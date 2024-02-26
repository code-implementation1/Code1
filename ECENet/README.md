# ECENet

paper link: http://arxiv.org/abs/2308.12894

## Requirements

```bash
conda create -n ms2.0.0 python=3.7
conda activate ms2.0.0
export MS_VERSION=2.0.0
pip install -r requirements.txt
```

## Inference

* ckpt url: https://pan.baidu.com/s/1BWJi5jN5oInNCcw6K_uPag?pwd=noah

* Running using SwinT-Base, and ADEChallengeData2016

`python main.py ./ckpt/sdt_swin-base-4-12_640x640_ade20k_1236length_ms_total-ms.py ./ckpt/ms_convert.ckpt ./ADEChallengeData2016 --eval mIoU mFscore mDice`

* Results

|aAcc|mIoU|mAcc|mFscore|mPrecision|mRecall|mDice|
|-|-|-|-|-|-|-|
|84.57|54.17|65.46|67.68|72.03|65.46|67.68|

## Citation

```txt
@inproceedings{liu2023boosting,
  title={Boosting Semantic Segmentation from the Perspective of Explicit Class Embeddings},
  author={Liu, Yuhe and Liu, Chuanjian and Han, Kai and Tang, Quan and Qin, Zengchang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={821--831},
  year={2023}
}
```