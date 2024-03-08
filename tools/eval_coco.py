from pycocotools.coco import COCO
import pycocotools
from mmdet.datasets.cocoeval_wrappers import COCOEvalWrapper, COCOEvalXclassWrapper

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
VOC_CLASSES = (
               'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
               'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
               'train', 'tv')
NONVOC_CLASSES = (
               'truck', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench',
               'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake',
               'bed', 'toilet', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

ann_file = 'path/to/instances_val2017.json'
det_file = 'path/to/coco_instancesDet_results.json'
coco_gt = COCO(ann_file)
coco_dt = coco_gt.loadRes(det_file)


pascal_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
    'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
    'train', 'tv']
pascal_ids = [CLASSES.index(name_cat) for name_cat in pascal_classes]

eval_cat_ids = coco_gt.get_cat_ids(cat_names=NONVOC_CLASSES)
pascal_correct_ids = coco_gt.get_cat_ids(cat_names=VOC_CLASSES)
import pdb;pdb.set_trace()
for idx, ann in enumerate(coco_gt.dataset['annotations']):
    if ann['category_id'] not in pascal_ids:
        coco_gt.dataset['annotations'][idx]['ignored_split'] = 0
    else:
        coco_gt.dataset['annotations'][idx]['ignored_split'] = 1

cocoEval = COCOEvalXclassWrapper(coco_gt, coco_dt, 'bbox')
cocoEval.params.useCats = 0
cocoEval.params.maxDets = (10, 20, 30, 50, 100, 300)

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()