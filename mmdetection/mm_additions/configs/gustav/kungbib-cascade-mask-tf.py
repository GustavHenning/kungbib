#from ... import TextFeatures
# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-tf.py'

TRAIN_TEST_VALID_FOLDERS="/data/gustav/datalab_data/poly-dn-2010-2020-720/"

# Adds more epochs as per 2x_coco
lr_config = dict(step=[16, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('News Article', 'Ad',)
data = dict(
    max_epochs=30,
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

# We can use the pre-trained Mask RCNN model to obtain higher performance TODO load_from a different model?
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
