_base_ = [
    'mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='./resnet50.pth',
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head = dict(
            num_classes=1,
        ),
    ),
)
# Dataset
dataset_type = 'CocoSplitDataset'
data_root = '/cluster_public_data/coco/origin/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)], 
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        type=dataset_type,
        pipeline=train_pipeline,
        ),
    val=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        type=dataset_type,
        ),
    test=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        type=dataset_type,
        ))

# Evaluation Hook. Priority 50
evaluation = dict(  
    interval=1,  # 验证的间隔。
    metric_items=['mAP', 'AR@10', 'AR@100'],
    metric=['bbox', 'segm'])

# EMAHook. Priority 49. Higher than evaluation hook
# custom_imports = dict(
#     imports=['mmdet.core.evaluation.ema_lr_hook'],
#     allow_failed_imports=False)
# custom_hooks=[
#     dict(
#         type='EMALrHook',
#         priority=49),
#     ]

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='/job_tboard')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir='./work_dirs/mask_rcnn/'