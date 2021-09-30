# The new config inherits a base config to highlight the necessary modification
_base_ = './kungbib-cascade-mask.py'

TRAIN_TEST_VALID_FOLDERS="/data/gustav/datalab_data/poly-dn-2010-2020-720/"
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

classes = ('News Article',)
data = dict(
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