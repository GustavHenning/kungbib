# TODO implement kb vanilla with 3 channels
# TODO implement kb ndim with 3 + n channels
# [x] make a script for downloading and verifying text corresponding to image. TODO Support different methods of vectorization
# TODO design architecture so experiments can be run using 
# a) vanilla or ndim(n=x), 
# b) different datasets, 
# c) output logs and corresponding visualizations to a directory (modify model.py to return history then https://valueml.com/how-to-visualize-the-training-process-in-keras/)
# based on config (a, b, c)

# Subtasks
# TODO verify that annotation format equals the one from coco. maybe visualize?

import os, sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library



# Directory to save logs and trained model
MODEL_NAME = "kb_vanilla"
MODEL_DIR = os.path.join(ROOT_DIR, "logs/" + MODEL_NAME)

from newspaper_config import NewspaperConfig
from newspaper_dataset import NewspaperDataset 

config = NewspaperConfig(dimensions=3)
dataset = NewspaperDataset(folder_path="/data/gustav/datalab_data/poly-dn-2010-2020-720/")