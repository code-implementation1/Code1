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


from collections import OrderedDict
import numpy as np
from prettytable import PrettyTable
from utils_mmcv import intersect_and_union, pre_eval_to_metrics

IGNORE_INDEX = 255
CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')
num_classes = len(CLASSES)

def pre_eval(pred, label):
    """
    evaluation
    """
    return intersect_and_union(
                pred,
                label,
                150,
                IGNORE_INDEX,
                # as the labels has been converted when dataset initialized
                # in `get_palette_for_custom_classes ` this `label_map`
                # should be `dict()`, see
                # https://github.com/open-mmlab/mmsegmentation/issues/1415
                # for more ditails
                label_map=dict(),
                reduce_zero_label=True)

def evaluate(results, metric='mIoU'):
    """Evaluate the dataset.

    Args:
        results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                results or predict segmentation map for computing evaluation
                metric.
        metric (str | list[str]): Metrics to be evaluated. 'mIoU',
            'mDice' and 'mFscore' are supported.
        logger (logging.Logger | None | str): Logger used for printing
            related information during evaluation. Default: None.
        gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
            used in ConcatDataset

    Returns:
        dict[str, float]: Default metrics.
    """

    if isinstance(metric, str):
        metric = [metric]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metric).issubset(set(allowed_metrics)):
        raise KeyError('metric {} is not supported'.format(metric))

    eval_results = {}
    # test a list of files
    ret_metrics = pre_eval_to_metrics(results, metric)

    # Because dataset.CLASSES is required for per-eval.
    if CLASSES is None:
        class_names = tuple(range(num_classes))
    else:
        class_names = CLASSES

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print('per class results:')
    print('\n' + class_table_data.get_string())
    print('Summary:')
    print('\n' + summary_table_data.get_string())

    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value / 100.0
        else:
            eval_results['m' + key] = value / 100.0

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx] / 100.0
            for idx, name in enumerate(class_names)
        })

    return eval_results
