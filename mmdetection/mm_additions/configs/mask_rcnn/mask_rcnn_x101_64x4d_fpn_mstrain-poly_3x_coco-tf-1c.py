_base_ = [
    '../common/mstrain-poly_3x_coco_instance-tf-1c.py',
    '../_base_/models/mask_rcnn_r50_fpn-tf.py'
]

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=None#dict(
            #type='Pretrained',
            #checkpoint='open-mmlab://detectron2/resnext101_32x8d')
            ))
