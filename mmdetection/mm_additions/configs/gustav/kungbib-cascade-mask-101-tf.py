_base_ = '../mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco-tf.py'
        
NUM_CLASSES=6

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES),
        mask_head=dict(num_classes=NUM_CLASSES)))

load_from = 'checkpoints/resnet101_caffe-3ad79236.pth'
