from mmdet.datasets import PIPELINES
import sys, json
from pprint import pprint
import numpy as np
from .encoders.bert import BERT
from .encoders.doc_to_vec import Doc2Vec
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_CHANNELS = 3
DEBUG_IMAGE = False


@PIPELINES.register_module()
class TextFeatures:
    def __init__(self,
                dimensions=3,
                encoder="doc2vec",
                model_name="multi-qa-MiniLM-L6-cos-v1"): # only for bert
        self.dimensions = dimensions
        print("encoder is {} with {} dimensions and model_name {}".format(encoder, dimensions, model_name))
        if encoder == "bert":
            self.encoder = BERT(dimensions=dimensions, model_name=model_name)
        elif encoder == "doc2vec":
            self.encoder = Doc2Vec(dimensions, model_name=model_name)
        else:
            print("Unrecognized text features encoder {}, using BERT instead.".format(encoder))
            self.encoder = BERT(dimensions, model_name=model_name)
        

    def __call__(self, results):

        if DEBUG_IMAGE:
            fig, ax = self.draw_base_image(results)

        scaleX = results["img_shape"][0] / results["ori_shape"][0]
        scaleY = results["img_shape"][1] / results["ori_shape"][1]

        pad_h = int(np.ceil(results["img"].shape[0] / results["pad_size_divisor"])) * results["pad_size_divisor"]
        pad_w = int(np.ceil(results["img"].shape[1] / results["pad_size_divisor"])) * results["pad_size_divisor"]
        if DEBUG_IMAGE:
            #print(results)
            print("size of img is {}, pad_h, pad_w is {} {}, ori_shape is {}".format(results["img_shape"], pad_h, pad_w, results["ori_shape"]))
            
        text_feature_array = np.zeros((pad_h, pad_w, 
            self.dimensions), dtype=np.float32)

        texts = self.get_text_json(results["filename"])
        # for each text block, encode it and stick it into the text_feature_array between x, y, width and height.
        # TODO how do we handle set dimensional sizes from the encoders?
        for block in texts["content"]:
            text = block["text"]
            encoded_vector = self.encoder.encode(text)
            x = int(block["x"] * scaleX) if results["flip_direction"] != 'horizontal' else pad_w - int(block["x"] * scaleX) - int(block["width"] * scaleX) 
            y = int(block["y"] * scaleY) 
            w = int(block["width"] * scaleX) 
            h = int(block["height"] * scaleY)
            text_feature_array[x:w][y:h] = encoded_vector
            if DEBUG_IMAGE:
                self.draw_rect(ax,x,y,w,h)

        self.show_plot()
        rgbtf = np.concatenate((results["img"], text_feature_array), axis=2)
        results["img_rgb"] = results["img"]
        results["img"] = rgbtf
        # set image shape
        results["img_shape"] = results["img_shape"][0], results["img_shape"][1], BASE_CHANNELS + self.dimensions

        return results
    
    def get_text_json(self, image_file_path):
        text_file_path = image_file_path.replace("/images/", "/text/").replace(".jpg", ".json")
        with open(text_file_path, encoding='utf-8') as fh:
            data = json.load(fh)
            return data

    def draw_base_image(self, results):
        if not DEBUG_IMAGE:
            return 
        fig, ax = plt.subplots()
        ax.imshow(results["img"])
        return fig, ax

    def draw_rect(self, ax, x, y, w, h):
        if not DEBUG_IMAGE:
            return 
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)  

    def show_plot(self):
        if not DEBUG_IMAGE:
            return 
        plt.show()
        
@PIPELINES.register_module()
class RemoveTextFeatures:
    def __call__(self, results):
        results["img"] = results["img"][:,:,0:3]