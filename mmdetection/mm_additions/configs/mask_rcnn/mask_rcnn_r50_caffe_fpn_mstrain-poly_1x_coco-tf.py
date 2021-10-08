_base_ = './mask_rcnn_r50_fpn_1x_coco-tf.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=None#dict(
            #type='Pretrained',
            #checkpoint='open-mmlab://detectron2/resnet50_caffe')
            ))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False, extra_dims=384) # TODO make extra_dims a cfg option
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Normalize', **img_norm_cfg), #TODO in place cv2.subtract doesnt work well with additional channels.
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='TextFeatures', dimensions=3, encoder="", model_name=""), # TODO this should perhaps be after resize
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
           # dict(type='RemoveTextFeatures'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='TextFeatures', dimensions=3, encoder="", model_name=""),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']), 
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
