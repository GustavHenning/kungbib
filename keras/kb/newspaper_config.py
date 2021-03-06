import numpy as np

import os, sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config

class NewspaperConfig(Config):

    def __init__(self, dimensions):
        TF_DIMS = dimensions
        # the ndim special
        normal_mean_img = np.array([123.7, 116.8, 103.9])
        IMAGE_CHANNEL_COUNT = 3 + dimensions if dimensions > 3 else 3
        MEAN_PIXEL = np.concatenate([normal_mean_img, np.repeat(0,dimensions)]) if dimensions > 3 else normal_mean_img
    """Configuration for training on the newspaper dataset
    """
    # Give the configuration a recognizable name
    NAME = "newspaper"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

