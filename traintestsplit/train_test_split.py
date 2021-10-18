import json, sys
import random
from collections import Counter

expected_class_order = [{'id': 0, 'name': 'News Article'}, {'id': 1, 'name': 'Ad'}, {'id': 2, 'name': 'Listing'}, {'id': 3, 'name': 'Weather'}, {'id': 4, 'name': 'Death'}, {'id': 5, 'name': 'Crossword'}]

def arrayify(categories):
    cats = [None] * len(categories)
    for cat in categories:
        cats[cat['id']] = cat['name']
    return cats

def same_order(data):
    if len(data["categories"]) != len(expected_class_order):
        print(data["categories"])
        print("categories are not the same as expected class order!")
        print(expected_class_order)
        sys.exit(1)
    for cat in data["categories"]:
        for exp in expected_class_order:
            if cat["id"] == exp["id"] and cat['name'] != exp['name']:
                return False
    return True

def stringify_categories(data):
    cats = arrayify(data["categories"])
    for ann in data["annotations"]:
        ann["category_id"] = cats[ann["category_id"]]

def index_categories(data, categories):
    cats = arrayify(categories)
    for ann in data["annotations"]:
        ann["category_id"] = cats.index(ann["category_id"])

def verify_or_modify_class_order(data):
    if same_order(data):
        return data
    stringify_categories(data)
    index_categories(data, expected_class_order)
    data["categories"] = expected_class_order
    return data

# TODO always take the latest segmentation from rekt, currently have a problem where multiple annotations are exported when re-labeling... 
def filter_empty(annotation):
    return len(annotation['segmentation']) > 0

def create_annotations(dataset_folder_path, annotations_prefix, images, data):
    indices = [index["id"] for index in images]

    annotations = list(
        filter(lambda annotation: annotation["image_id"] in indices, data["annotations"])
    )
    annotations = list(filter(filter_empty, annotations))

    data = {
        "images": images,
        "categories": data["categories"],
        "annotations": annotations,
        "info": data["info"],
    }
    data = verify_or_modify_class_order(data)
    
    cats = arrayify(data["categories"])
    cat_count = [cats[item['category_id']] for item in data["annotations"]]
    print("{} has this class distribution: {} ".format(annotations_prefix, Counter(cat_count)))

    # os.makedirs(f"{destination_folder}/images", exist_ok=True)
    file_name = dataset_folder_path + "/" + annotations_prefix + "_annotations.json"
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

def train_val_split(dataset_folder_path, json_filename, train_prop=0.8, test_prop=None, seed=None):
    """
    Create random train, test and optionally validation split from a COCO annotation json file.

    Args:
        json_filename (str): Path and filename to coco json file with annotations.
        train_prop (float): Proportion of observations in training set.
        test_prop (None | float): Proportion of observations in test set. If None is supplied, only a validation set will be created with 1 - train_prop size.
        seed (None | int): Set an optional random seed. 
    """
    if test_prop is not None:
        if sum([train_prop, test_prop]) >= 1.0:
            print("train_prop [%d] and test_prop [%d] cannot be larger than or equal to 1.0, no validation set will be created. ".format(train_prop, test_prop))
            sys.exit(1)

    with open(dataset_folder_path + "/" + json_filename) as f:
        data = json.load(f)

    nr_obs = len(data["images"])

    if seed:
        random.seed(seed)
    
    if test_prop is not None: # train, test and validation
        train_images = random.sample(data["images"], k=int(nr_obs * train_prop))
        rest_images = {"images" : [image for image in data["images"] if image not in train_images]}

        test_images = random.sample(rest_images["images"], k=int(nr_obs * test_prop))
        valid_images = [image for image in data["images"] if image not in train_images and image not in test_images]

        create_annotations(dataset_folder_path, "train", train_images, data)
        create_annotations(dataset_folder_path, "test", test_images, data)
        create_annotations(dataset_folder_path, "valid", valid_images, data)
    else: # only train and validation
        train_images = random.sample(data["images"], k=int(nr_obs * train_prop))
        valid_images = [image for image in data["images"] if image not in train_images]

        create_annotations(dataset_folder_path, "train", train_images, data)
        create_annotations(dataset_folder_path, "valid", valid_images, data)

DATASET_FOLDER="poly-729/"

train_val_split(dataset_folder_path=DATASET_FOLDER, json_filename="result.json", train_prop=0.5, test_prop=0.25, seed=5)