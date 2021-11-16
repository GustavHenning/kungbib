_base_ = [
    '../common/mstrain-poly_3x_coco_instance-tf.py',
    '../_base_/models/mask_rcnn_r50_fpn-tf.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=None #dict(type='Pretrained',
                 #     checkpoint='torchvision://resnet101')
                      ))
