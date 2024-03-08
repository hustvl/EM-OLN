_base_ = [
    'mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='resnet50.pth',
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
data_root = '/horizon-bucket/aidi_public_data/coco/origin/'

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
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
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
        pipeline=val_pipeline,),
    test=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        ann_file=data_root + 'annotations/instances_val2017.json',
        # ann_file='/data/cheng03.wang/workspace/repository/object_localization_network/work_dirs/local_coco_val.json',
        img_prefix=data_root + 'val2017/',
        type=dataset_type,
        pipeline=val_pipeline))

# Evaluation Hook. Priority 50
evaluation = dict(  
    interval=1,  # 验证的间隔。,
    metric_items=['mAP', 'AR@10', 'AR@100'],
    metric=['bbox', 'segm'])

# EMAHook. Priority 49. Higher than evaluation hook
# custom_imports = dict(
#     imports=['mmdet.core.evaluation.ema_lr_hook',
#              'mmdet.core.evaluation.tensorboard_hook_custom',
#     ],
#     allow_failed_imports=False)

# custom_hooks=[
#     dict(
#         type='EMALrHook',
#         resume_from='/data/cheng03.wang/workspace/repository/trained_weights/mask_rcnn/maskrcnn-rpnMatchLowQualityThresh0-frosenBN-EMA1e-3.pth',
#         priority=49),
#     ]

# Load checkpoint
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])

work_dir='./work_dirs/mask_rcnn/'