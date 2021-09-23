
from newspaper_config import NewspaperConfig
import os, sys
import numpy as np
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import skimage, json

class NewspaperDataset(utils.Dataset):
    
    def load_newspaper(self, config, dataset_dir, dataset_type="train", class_ids=None):
        """Load a kb newspaper dataset
        dataset_type: "train", "test" or "valid". Corresponding annotations should exist within the dataset_dir.
        class_ids: If provided, only loads images that have the given classes.
        """
        image_dir = dataset_dir + "/images"
        text_dir = dataset_dir + "/text"

        data = {}
        with open(os.path.join(dataset_dir, dataset_type + "_annotations.json")) as json_file:
             data = json.load(json_file)
        if len(data.keys()) == 0:
            print("Cant find any data within dataset_dir, dataset_type {} {}".format(dataset_dir, dataset_type))
        
        # Add classes
        # TODO Load all classes or a subset?

        categories = data["categories"] 
        self.add_class(config.NAME, 0, "BG")
        for cat in categories:
            self.add_class(config.NAME, cat["id"] + 1, cat["name"]) # Real categories start at 1

        image_info = {}
        for info in data["images"]:
            image_info[info["id"]] = {}
            image_info[info["id"]]["width"] = info["width"]
            image_info[info["id"]]["height"] = info["height"]
            image_info[info["id"]]["file_name"] = info["file_name"]
        # append annotations for ez
        for ann in data["annotations"]:
            if not "annotations" in image_info[ann["image_id"]]:
                image_info[ann["image_id"]]["annotations"] = []
            image_info[ann["image_id"]]["annotations"].append(ann)
        # Add images
        for id in image_info:
            self.add_image(
                config.NAME, image_id=id,
                path=os.path.join(dataset_dir, image_info[id]["file_name"]),
                width=image_info[id]["width"],
                height=image_info[id]["height"],
                annotations=image_info[id]["annotations"] # TODO coco uses iscrowd == false, problem?
            )
    
    def load_mask(self, image_id):
        """
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = annotation["category_id"] + 1 # bg is 0.
            m = self.annToMask(annotation, image_info["height"], image_info["width"])
            # Note: some arbitrary stuff is skipped here that is checked for is coco
            instance_masks.append(m)
            class_ids.append(class_id)
        
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(NewspaperDataset, self).load_mask(image_id)
    
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        
        ## TODO add support for multidimensional text

        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if "path" in info:
            return info["path"] # TODO maybe append file:///?
        else:
            super(NewspaperDataset, self).image_reference(image_id)

#for testing
if __name__ == '__main__': 
    config = NewspaperConfig(dimensions=3)
    NewspaperDataset().load_newspaper(config, "/data/gustav/datalab_data/poly-dn-2010-2020-720/")