_base_ = '../mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco-tf.py'
    
NUM_CLASSES=6

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES),
        mask_head=dict(num_classes=NUM_CLASSES)))
        
load_from = 'checkpoints/resnext101_32x8d-1516f1aa.pth'

