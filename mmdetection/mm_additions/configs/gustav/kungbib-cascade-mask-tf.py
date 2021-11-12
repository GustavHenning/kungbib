_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-tf.py'

NUM_CLASSES=6

MAX_EPOCHS=32
EVAL_INTERVAL=1

LEARNING_RATE=0.005
MOMENTUM=0.9
WEIGHT_DECAY=0.0001

dataset_type = 'COCODataset'
classes = ('News Unit', 'Advertisement', 'Listing', 'Weather', 'Death Notice', 'Game',)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=NUM_CLASSES),
        mask_head=dict(num_classes=NUM_CLASSES)))

evaluation = dict(interval=EVAL_INTERVAL)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False, extra_dims=3) # TODO make extra_dims a cfg option

IN_SET_FOLDERS="/data/gustav/datalab_data/model/dn-2010-2020/"
NEAR_SET_FOLDERS="/data/gustav/datalab_data/model/dn-svd-2001-2004/"
OUT_SET_FOLDERS="/data/gustav/datalab_data/model/ab-ex-2001-2004/"

data = dict(
    max_epochs=MAX_EPOCHS,
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=10,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix=IN_SET_FOLDERS,
        classes=classes,
        ann_file=IN_SET_FOLDERS + '/train_annotations.json'),
    val=dict(img_prefix=[IN_SET_FOLDERS, NEAR_SET_FOLDERS, OUT_SET_FOLDERS],
                classes=classes,
                ann_file=[IN_SET_FOLDERS + '/valid_annotations.json', NEAR_SET_FOLDERS + '/valid_annotations.json', OUT_SET_FOLDERS + '/valid_annotations.json']),
    test=dict(
        img_prefix=IN_SET_FOLDERS,
        classes=classes,
        ann_file=IN_SET_FOLDERS + '/test_annotations.json'))

optimizer = dict(type='SGD', lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) # 0.0025 * samples_per_gpu
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(type='EpochBasedRunner', max_epochs=MAX_EPOCHS)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9,11])

# We can use the pre-trained Mask RCNN model to obtain higher performance TODO load_from a different model?
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
