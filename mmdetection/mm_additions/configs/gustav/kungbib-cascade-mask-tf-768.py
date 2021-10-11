_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-tf-768.py'

TRAIN_TEST_VALID_FOLDERS="/data/gustav/datalab_data/poly-dn-2010-2020-720/"
MAX_EPOCHS=36

# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=MAX_EPOCHS)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('News Article', 'Ad',)
data = dict(
    max_epochs=MAX_EPOCHS,
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=8,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix=TRAIN_TEST_VALID_FOLDERS,
        classes=classes,
        ann_file=TRAIN_TEST_VALID_FOLDERS + '/train_annotations.json'),
    val=dict(
        img_prefix=TRAIN_TEST_VALID_FOLDERS,
        classes=classes,
        ann_file=TRAIN_TEST_VALID_FOLDERS + '/valid_annotations.json'),
    test=dict(
        img_prefix=TRAIN_TEST_VALID_FOLDERS,
        classes=classes,
        ann_file=TRAIN_TEST_VALID_FOLDERS + '/test_annotations.json'))


optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])


# We can use the pre-trained Mask RCNN model to obtain higher performance 
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
