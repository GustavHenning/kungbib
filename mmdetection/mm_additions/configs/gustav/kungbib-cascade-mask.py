_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py'

NUM_CLASSES=6

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES),
        mask_head=dict(num_classes=NUM_CLASSES)))
        
load_from = 'checkpoints/full/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'
