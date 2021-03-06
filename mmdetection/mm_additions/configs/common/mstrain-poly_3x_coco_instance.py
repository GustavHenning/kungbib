_base_ = '../_base_/default_runtime.py'

dataset_type = 'CocoDataset'
data_root = "/data/gustav/datalab_data/model/dn-2010-2020/"

MAX_EPOCHS=16
EVAL_INTERVAL=1
REPEAT_TRAIN=3

LEARNING_RATE=0.0025
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
STEP_LENGTH=8

classes = ('News Unit', 'Advertisement', 'Listing', 'Weather', 'Death Notice', 'Game',)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False,
    extra_dims=0)
    
evaluation = dict(interval=EVAL_INTERVAL, metric=['bbox', 'segm'])


# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
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


IN_SET_FOLDERS="/data/gustav/datalab_data/model/dn-2010-2020/"
NEAR_SET_FOLDERS="/data/gustav/datalab_data/model/dn-svd-2001-2004/"
OUT_SET_FOLDERS="/data/gustav/datalab_data/model/ab-ex-2001-2004/"


data = dict(
    max_epochs=MAX_EPOCHS,
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=12,  # Worker to pre-fetch data for each single GPU
    train=dict(    
        type='RepeatDataset',
        times=REPEAT_TRAIN,
        dataset=dict(
            type=dataset_type,
            img_prefix=IN_SET_FOLDERS,
            classes=classes,
            ann_file=IN_SET_FOLDERS + '/train_annotations.json',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        img_prefix=IN_SET_FOLDERS,
        classes=classes,
        ann_file=IN_SET_FOLDERS + '/valid_annotations.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=[IN_SET_FOLDERS, NEAR_SET_FOLDERS, OUT_SET_FOLDERS],
        classes=classes,
        ann_file=[IN_SET_FOLDERS + '/test_annotations.json', NEAR_SET_FOLDERS + '/test_annotations.json', OUT_SET_FOLDERS + '/test_annotations.json'],
        pipeline=test_pipeline))



optimizer = dict(type='SGD', lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) # 0.0025 * samples_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(type='EpochBasedRunner', max_epochs=MAX_EPOCHS)
# learning policy https://arxiv.org/pdf/1506.01186.pdf

lr_config = dict(  
    policy='step',  
    warmup='linear',  
    warmup_iters=500, 
    warmup_ratio=0.001,
    gamma=0.2,  
    step=STEP_LENGTH) # lr divided by 5 every 8 epochs
