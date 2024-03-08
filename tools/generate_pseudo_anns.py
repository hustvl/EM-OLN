import json
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


PASCAL_IDS = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]

def gen_pseudo_ann_nonVOC(gt_coco, dt_coco, output_dir, unseen_thresh=0.3, select_thresh=0.75):

    pseudo_ann = []
    for imgid, ann in gt_coco.imgToAnns.items():
        dt = dt_coco.imgToAnns[imgid]
        ann_voc_bboxes = np.array([ele['bbox'] for ele in ann if ele['category_id'] in PASCAL_IDS])
        if len(ann_voc_bboxes)==0:
            continue
        ann_voc_bboxes[:, 2:] = ann_voc_bboxes[:, :2] + ann_voc_bboxes[:, 2:]

        dt_bboxes = np.array([ele['bbox'] for ele in dt])
        dt_scores = np.array([ele['score'] for ele in dt])
        dt_bboxes[:, 2:] = dt_bboxes[:, :2] + dt_bboxes[:, 2:]
        
        overlaps = bbox_overlaps(ann_voc_bboxes, dt_bboxes)
        max_overlaps = overlaps.max(axis=0)

        unseen_bboxes_index = np.nonzero(max_overlaps<unseen_thresh)[0]
        unseen_bboxes = dt_bboxes[unseen_bboxes_index]
        unseen_scores = dt_scores[unseen_bboxes_index]
        unseen_bboxes_index = np.nonzero(unseen_scores>select_thresh)[0]
        if len(unseen_bboxes_index)==0:
            continue
        unseen_bboxes = unseen_bboxes[unseen_bboxes_index]
        unseen_scores = unseen_scores[unseen_bboxes_index]

        unseen_bboxes[:, 2:] = unseen_bboxes[:, 2:] - unseen_bboxes[:, :2]
        area = unseen_bboxes[:, 2] * unseen_bboxes[:, 3]
        for i in range(len(unseen_bboxes_index)):
            ann = {'area':area[i],
                   'bbox':unseen_bboxes[i].tolist(), 
                   'category_id': 1000, 
                   'image_id': imgid, 
                   'iscrowd': 0, 
                   'pseudo_score': unseen_scores[i],
                   'id': 10000000+len(pseudo_ann)}
            pseudo_ann.append(ann)
    print(f'Totally {len(pseudo_ann)} pseudo anns.')
    output_file = f'{output_dir}/pseduo_anns_{select_thresh}_{len(pseudo_ann)}_nonVOC.json'
    with open(output_file, 'w') as f:
        json.dump({'annotations':pseudo_ann}, f)

    return output_file

def gen_pseudo_ann_COCOAll(gt_coco, dt_coco, output_dir, unseen_thresh=0.3, select_thresh=0.75):

    pseudo_ann = []
    for imgid, ann in gt_coco.imgToAnns.items():
        dt = dt_coco.imgToAnns[imgid]
        ann_bboxes = np.array([ele['bbox'] for ele in ann])
        if len(ann_bboxes)==0:
            continue
        ann_bboxes[:, 2:] = ann_bboxes[:, :2] + ann_bboxes[:, 2:]

        dt_bboxes = np.array([ele['bbox'] for ele in dt])
        dt_scores = np.array([ele['score'] for ele in dt])
        dt_bboxes[:, 2:] = dt_bboxes[:, :2] + dt_bboxes[:, 2:]
        
        overlaps = bbox_overlaps(ann_bboxes, dt_bboxes)
        max_overlaps = overlaps.max(axis=0)

        unseen_bboxes_index = np.nonzero(max_overlaps<unseen_thresh)[0]
        unseen_bboxes = dt_bboxes[unseen_bboxes_index]
        unseen_scores = dt_scores[unseen_bboxes_index]
        unseen_bboxes_index = np.nonzero(unseen_scores>select_thresh)[0]
        if len(unseen_bboxes_index)==0:
            continue
        unseen_bboxes = unseen_bboxes[unseen_bboxes_index]
        unseen_scores = unseen_scores[unseen_bboxes_index]

        unseen_bboxes[:, 2:] = unseen_bboxes[:, 2:] - unseen_bboxes[:, :2]
        area = unseen_bboxes[:, 2] * unseen_bboxes[:, 3]
        for i in range(len(unseen_bboxes_index)):
            ann = {'area':area[i],
                   'bbox':unseen_bboxes[i].tolist(), 
                   'category_id': 1000, 
                   'image_id': imgid, 
                   'iscrowd': 0, 
                   'pseudo_score': unseen_scores[i],
                   'id': 10000000+len(pseudo_ann)}
            pseudo_ann.append(ann)
    print(f'Totally {len(pseudo_ann)} pseudo anns.')
    output_file = f'{output_dir}/pseduo_anns_{select_thresh}_{len(pseudo_ann)}_allCOCO.json'
    with open(output_file, 'w') as f:
        json.dump({'annotations':pseudo_ann}, f)

    return

if __name__=="__main__":
    for_all_coco = True
    coco_gt_path = "path/to/coco/annotaions/instances_train2017.json"
    if for_all_coco:
        coco_det_path = "path/to/coco/detections/allCOCO.bbox.json"
    else:
        coco_det_path = "path/to/coco/detections/nonVOC.bbox.json"
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # generate pseudo anns
    if for_all_coco:
        # for COCO to others scenarios
        gt_coco = COCO(coco_gt_path)
        dt_coco = gt_coco.loadRes(coco_det_path)
        pseudo_ann_file = gen_pseudo_ann_COCOAll(gt_coco, dt_coco, output_dir, select_thresh=0.87)
    else:
        gt_coco = COCO(coco_gt_path)
        dt_coco = gt_coco.loadRes(coco_det_path)
        # for VOC to non-VOC scenario
        pseudo_ann_file = gen_pseudo_ann_nonVOC(gt_coco, dt_coco, output_dir, select_thresh=0.80)

    # merge with original COCO annotations
    pseudo_tag = os.path.basename(pseudo_ann_file).split('_', 1)[-1]
    pseudo_ann = json.load(open(pseudo_ann_file, 'r'))
    pseudo_ann = pseudo_ann['annotations']

    coco_train_ann = json.load(open(coco_gt_path, 'r'))
    coco_train_ann['annotations'].extend(pseudo_ann)
    print(f'coco instances_train2017_withPseudoBox number: {len(coco_train_ann["annotations"])}')

    if for_all_coco:
        final_file = f'{output_dir}/instances_train2017_withPseudoBox_allCOCO.json'
    else:
        final_file = f'{output_dir}/instances_train2017_withPseudoBox_nonVOC.json'
    with open(final_file, 'w') as f:
        json.dump(coco_train_ann, f)