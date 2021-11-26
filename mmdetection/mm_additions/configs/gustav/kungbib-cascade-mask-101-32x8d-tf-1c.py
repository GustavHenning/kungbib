_base_ = '../mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco-tf-1c.py'
    
NUM_CLASSES=1

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES),
        mask_head=dict(num_classes=NUM_CLASSES)))

load_from = 'checkpoints/full/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042-8bd2c639.pth'

