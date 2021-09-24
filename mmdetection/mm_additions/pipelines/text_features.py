from mmdet.datasets import PIPELINES
import sys, json
from pprint import pprint
import numpy as np
from .encoders.bert import BERT
from .encoders.doc_to_vec import Doc2Vec


BASE_CHANNELS=3

@PIPELINES.register_module()
class TextFeatures:
    def __init__(self,
                dimensions=3,
                encoder="doc2vec"):
        self.dimensions = dimensions
        if encoder is "bert":
            self.encoder = BERT(dimensions)
        elif encoder is "doc2vec":
            self.encoder = Doc2Vec(dimensions)
        else:
            print("Unrecognized text features encoder {}, using BERT instead.".format(encoder))
            self.encoder = BERT(dimensions)
        

    def __call__(self, results):
        text_feature_array = np.zeros((results["img_shape"][0], results["img_shape"][1], 
            self.dimensions), dtype=np.float32)

        texts = self.get_text_json(results["filename"])
        # for each text block, encode it and stick it into the text_feature_array between x, y, width and height.
        # TODO how do we handle set dimensional sizes from the encoders?
        for block in texts["content"]:
            text = block["text"]
            encoded_vector = self.encoder.encode(text)
            text_feature_array[block["x"]:block["width"]][block["y"]:block["height"]] = encoded_vector

        rgbtf = np.concatenate((results["img"], text_feature_array), axis=2)
        results["img"] = rgbtf
        
        # set image shape
        results["img_shape"] = results["img_shape"][0], results["img_shape"][1], BASE_CHANNELS + self.dimensions
        return results
    
    def get_text_json(self, image_file_path):
        text_file_path = image_file_path.replace("/images/", "/text/").replace(".jpg", ".json")
        with open(text_file_path, encoding='utf-8') as fh:
            data = json.load(fh)
            return data
