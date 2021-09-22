from mmdet.datasets import PIPELINES
import sys
from pprint import pprint
import numpy as np

BASE_CHANNELS=3
ADDITIONAL_CHANNELS=1

@PIPELINES.register_module()
class TextFeatures:

    def __call__(self, results):
        # TODO add random noise in an additional channel and see if it trains
        # TODO normalize needs to be adjusted for more channels
        # TODO RESNET input head needs to be adjusted 
        text_feature_array = np.zeros((results["img_shape"][0], results["img_shape"][1], ADDITIONAL_CHANNELS))
        # TODO Actually fill text_feature_array with noise or something useful
        rgbtf = np.concatenate((results["img"], text_feature_array), axis=2)
        results["img"] = rgbtf
        
        # set image shape
        results["img_shape"] = results["img_shape"][0], results["img_shape"][1], BASE_CHANNELS + ADDITIONAL_CHANNELS
        return results