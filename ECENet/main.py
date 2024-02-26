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

import argparse
from ade_dataset import ADE20KDataset
from config import Config
from metrics import pre_eval, evaluate
from sdt_head_1236length_total import SegDecodingTransformer
from swin import SwinTransformer

import mindspore
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F
import numpy as np
from tqdm import tqdm

class EncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(EncoderDecoder, self).__init__()
        self.backbone = SwinTransformer(**cfg.model.backbone)
        self.decode_head = SegDecodingTransformer(**cfg.model.decode_head)
        self.out_channels = 150
        self.index = 0

    def encode_decode(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x = self.backbone(img)
        out = self.decode_head.forward_test(x)
        out = F.interpolate(out, size=img.shape[2:], mode="bilinear")

        self.index += 1
        return out

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = 608, 608
        h_crop, w_crop = 640, 640
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = F.interpolate(preds, size=img_meta['ori_shape'][:2], mode="bilinear")
        return preds

    def inference(self, img, img_meta, rescale=True, test_mode="slide"):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert test_mode in ['slide', 'whole']
        if test_mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta['flip']
        if flip:
            flip_direction = img_meta['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def forward(self, imgs, img_metas, rescale=True):
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path', type=str)
    parser.add_argument('checkpoint', help='checkpoint file', type=str)
    parser.add_argument('data_root', help='ADEChallengeData2016', default="ADEChallengeData2016")
    parser.add_argument('--eval', help='eval metric', default="mIoU", type=str, nargs='+')
    args = parser.parse_args()

    config = args.config
    cfg = Config.fromfile(config)
    model = EncoderDecoder(cfg).set_train(False).cuda().eval()
    res = mindspore.load_checkpoint(args.checkpoint, model)

    dataset = ADE20KDataset(data_root=args.data_root, split="validation")

    preds = []
    labels = []
    for idx, (img, img_metas, label) in enumerate(tqdm(dataset)):
        for i in range(len(img)):
            img[i] = img[i].cuda().unsqueeze(0)
        pred = model(img, img_metas)
        preds.extend(pred)
        labels.append(np.array(label))

    results = []

    for pred, label in tqdm(zip(preds, labels)):
        results.append(pre_eval(pred, label))

    evaluate(results, args.eval)
