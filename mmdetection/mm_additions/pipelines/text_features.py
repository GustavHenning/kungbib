from mmdet.datasets import PIPELINES
import sys, json
from pprint import pprint
import numpy as np
from .encoders.bert import BERT
from .encoders.doc_to_vec import Doc2Vec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import preprocessing

from timeit import default_timer as timer


from pylab import *
from mpl_toolkits.mplot3d import Axes3D

BASE_CHANNELS = 3
DEBUG_IMAGE = False
SIMPLE_VIZ_WITH_TF = False
DEBUG_TIME = False
VISUALIZE_EMBEDDINGS = False

IMAGES_FOR_THESIS = ["images/dark-10120707_part01_page006_TFE3FSf.jpg",
"images/dark-10120964_part01_page005_FZKQ4Zg.jpg",
"images/dark-10120964_part01_page017_AVThDFz.jpg",
"images/dark-10120964_part01_page018_OD046mU.jpg",
"images/dark-10120964_part01_page032_y3LXnnL.jpg",
"images/dark-10366628_part01_page002_KZBgVFj.jpg",
"images/dark-10366628_part01_page021_BSsg1sL.jpg",
"images/dark-10366628_part02_page017_9rwJ51v.jpg",
"images/dark-4407198_part03_page015_7Zovctl.jpg",
"images/dark-4425041_part01_page001_JIefsDa.jpg",
"images/dark-4425337_part02_page051_GQU6vjW.jpg",
"images/dark-4993682_part01_page022_zk6UnuL.jpg",
"images/dark-5027440_part01_page028_Jd55Bvg.jpg",
"images/dark-6608804_part01_page011_TZqA3El.jpg",
"images/dark-6733161_part04_page033_GqEyiAA.jpg",
"images/dark-6745860_part04_page037_1RoKim0.jpg",
"images/dark-6748921_part04_page009_kvtooQ3.jpg",
"images/dark-6749981_part02_page031_PuwcAOD.jpg",
"images/dark-6745364_part05_page001_7vqUVUu.jpg",
"images/dark-5009648_part03_page001_axCUOek.jpg",
"images/dark-10120960_part01_page001_eL02GFP.jpg"]

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
        
        if DEBUG_TIME:
            start = timer()

        if self.should_show_debug_img(results):
            fig, ax = self.draw_base_image(results)
        

        scaleX = results["img_shape"][1] / results["ori_shape"][1]
        scaleY = results["img_shape"][0] / results["ori_shape"][0]

        img_postpad_w = results["img"].shape[1]
        img_postpad_h = results["img"].shape[0]

        img_prepad_w = results["img_shape"][1]
        img_prepad_h = results["img_shape"][0]

        if self.should_show_debug_img(results):
            #print(results)
            print(results["img_shape"])
            print(results["img"].shape)
            print("size of img is {}, postpad_w, postpad_h is {} {}, ori_shape is {}, scale is {} {}".format(results["img_shape"], img_postpad_w, img_postpad_h, results["ori_shape"], scaleX, scaleY))
        if DEBUG_TIME:    
            print("initialize array at {}".format(timer() - start))
        text_feature_array = np.zeros((img_postpad_h, img_postpad_w, 
            self.dimensions + 3), dtype=np.float32)
        if DEBUG_TIME:
            print("read text at {}".format(timer() - start))
        texts = self.get_text_json(results["filename"])
        if DEBUG_TIME:
            print("text read done at {}".format(timer() - start))
        # for each text block, encode it and stick it into the text_feature_array between x, y, width and height.
        # TODO how do we handle set dimensional sizes from the encoders?
        for block in texts["content"]:
            text = block["text"]
            #print(text)
            norm = self.encoder.encode(text) #self.normalize(self.encoder.encode(text))
            encoded_vector = np.pad(norm, (3,0), 'constant') # prepad rgb channels 

            total_width = img_postpad_w
            padding = img_postpad_w - img_prepad_w
            block_x = int(np.ceil(block["x"] * scaleX))

            w = int(np.ceil(block["width"] * scaleX)) 
            x = int(block["x"] * scaleX) if results["flip_direction"] != 'horizontal' else total_width - (padding + block_x + w)
            #print("total width: {}".format(total_width))
            #print("padding: {}".format(padding))
            #print("block_x: {}".format(block_x))
            #print("w_x: {}".format(w_x))
            y = int(block["y"] * scaleY) 
            h = int(np.ceil(block["height"] * scaleY))

            #for a in range(y, y+h):
            #    for b in range(x, x+w):
            #        text_feature_array[a][b] = encoded_vector
            text_feature_array[y:(y+h),x:(x+w),:] = encoded_vector
            if self.should_show_debug_img(results) and not SIMPLE_VIZ_WITH_TF:
                self.draw_rect(ax,x,y,w,h)
        if DEBUG_TIME:
            print("blocks done at {}".format(timer() - start))

        len_y = results["img"].shape[0]
        len_x = results["img"].shape[1]

        #self.confirm_empty(text_feature_array[:,:,0:3])
        text_feature_array[0:len_y,0:len_x,0:3] = results["img"]

        if DEBUG_TIME:
            print("concatenate done at {}".format(timer() - start))
        results["img_rgb"] = results["img"]
        results["img"] = text_feature_array

        if VISUALIZE_EMBEDDINGS and results["img_info"]["file_name"] in IMAGES_FOR_THESIS:
            self.visualize(results["img"], results)

        if self.should_show_debug_img(results) and SIMPLE_VIZ_WITH_TF:
            print(np.shape(results["img"][:,:,3:6]))
            ax.imshow(np.abs(self.normalize_3d_non_zero(results["img"][:,:,3:6])))


        if self.should_show_debug_img(results):

            plt.axis('off')
            fig.savefig("/home/gush/vis/{}/{}".format("tf" if SIMPLE_VIZ_WITH_TF else "normal", results["img_info"]["file_name"].split("_")[-1]), bbox_inches="tight")
            self.show_plot()

        if DEBUG_TIME:
            print("reassign at {}".format(timer() - start))
        # set image shape
        results["img_shape"] = results["img_shape"][0], results["img_shape"][1], BASE_CHANNELS + self.dimensions

        return results
    
    def should_show_debug_img(self, results):
        return DEBUG_IMAGE and results["img_info"]["file_name"] in IMAGES_FOR_THESIS

    def get_text_json(self, image_file_path):
        text_file_path = image_file_path.replace("/images/", "/text/").replace(".jpg", ".json")
        with open(text_file_path, encoding='utf-8') as fh:
            data = json.load(fh)
            return data

    def draw_base_image(self, results):
        if not DEBUG_IMAGE:
            return 
        fig, ax = plt.subplots(figsize=(10,10))
        
        fig.set_dpi(100)
        fig.canvas.manager.set_window_title(results["img_info"]["file_name"].split("/")[1])
        if not SIMPLE_VIZ_WITH_TF:
            ax.imshow(np.zeros(shape=np.shape(results["img"])))
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
    
    def visualize(self, m, results):
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection="3d")
        x, y = np.ogrid[0:m.shape[0], 0:m.shape[1]]
        img_colors=self.normalize_3d(m[:,:,0:3])

        ax.plot_surface(x, y, np.full((np.shape(x)[0], np.shape(y)[1]), 10.0, dtype=float), rstride=5, cstride=5, facecolors=(img_colors))
        tf_colors=np.abs(self.normalize_3d(m[:,:,3:6]))

        ax.plot_surface(x, y, np.full((np.shape(x)[0], np.shape(y)[1]), 0.0, dtype=float), rstride=5, cstride=5, facecolors=(tf_colors))
        plt.axis('off')
        fig.savefig("/home/gush/vis/3d/{}".format(results["img_info"]["file_name"].split("_")[-1]), bbox_inches="tight")
        plt.show()
        plt.pause(1000)

    def normalize(self, nparray):
        v_min = nparray.min(axis=0, keepdims=True) #axis=(0,1)
        v_max = nparray.max(axis=0, keepdims=True)
        return (nparray - v_min)/(v_max - v_min) if (v_max - v_min) != 0 else nparray

    def normalize_3d(self, nparray):
        v_min = nparray.min(axis=(0,1,2), keepdims=True) #axis=(0,1)
        v_max = nparray.max(axis=(0,1,2), keepdims=True)
        return (nparray - v_min)/(v_max - v_min) if (v_max - v_min) != 0 else nparray

    def normalize_3d(self, nparray):
        v_min = nparray.min(axis=(0,1,2), keepdims=True) #axis=(0,1)
        v_max = nparray.max(axis=(0,1,2), keepdims=True)
        return (nparray - v_min)/(v_max - v_min) if (v_max - v_min) != 0 else nparray


    def normalize_3d_non_zero(self, nparray):
        v_min = nparray.min(axis=(0,1,2), keepdims=True) #axis=(0,1)
        v_max = nparray.max(axis=(0,1,2), keepdims=True)
        result = (nparray - v_min)/(v_max - v_min) if (v_max - v_min) != 0 else nparray
        should_be_zero = (0.0 - v_min)/(v_max - v_min)

        np.subtract(should_be_zero, result, out=result, where=result==should_be_zero)
        return result
        return (nparray - v_min)/(v_max - v_min) if (v_max - v_min) != 0 else nparray

    def confirm_empty(self, arr):
        if(np.mean(arr) != 0.0):
            print("Expected empty array but found mean {}".format(np.mean(arr)))

@PIPELINES.register_module()
class RemoveTextFeatures:
    def __call__(self, results):
        results["img"] = results["img"][:,:,0:3]
        results["img_shape"] = results["img_shape"][0], results["img_shape"][1], 3
        return results