import json, sys, argparse
from pathlib import Path

def convert_to_poly(value):
    points = []
    points.append([value["x"], value["y"]])
    points.append([value["x"], value["y"] + value["height"]])
    points.append([value["x"] + value["width"], value["y"] + value["height"]])
    points.append([value["x"] + value["width"], value["y"]])
    return {"points" : points, "polygonlabels": value["rectanglelabels"]}

"""
    verify image_rotation = 0
    for each result:
        * make type from rectanglelabels to polygonlabels
        * make value x y w h -> points and rectanglelabels -> polygonlabels
        * project_id -> poly_project_id
"""
def relabel_rect(tar_poly_id, rect_indexed, img_id):
    rect_indexed[img_id]["project"] = int(tar_poly_id)
    rect_result = rect_indexed[img_id]["annotations"][0]["result"]
    poly = []
    for rect in rect_result:
        if rect["image_rotation"] != 0 or rect["value"]["rotation"] != 0:
            print("Image rotation is not 0 for %s. Exiting...".format(img_id))
            sys.exit(1)
        rect["type"] = "polygonlabels"
        rect["value"] = convert_to_poly(rect["value"])
        poly.append(rect)
    rect_indexed[img_id]["annotations"][0]["result"] = poly
    

def convert_rect_to_poly(tar_poly_id, rect_indexed):
    for img_id in rect_indexed:
        relabel_rect(tar_poly_id, rect_indexed, img_id)
    return rect_indexed

def verify_file_exists(path):
    file = Path(path)
    if not file.is_file():
        print("Could not find file: %s".format(path))
        sys.exit(1)

def label_studios_unique_id(image_name):
    return image_name.replace("_" + image_name.split("_")[-1].split(".")[0], "")

def examples_to_image_id_index(json_array):
    id_indexed = {}

    for example in json_array:
        if label_studios_unique_id(example["data"]["image"]) in id_indexed:
            print("File has been uploaded multiple times: %s. Exiting...".format(label_studios_unique_id(example["data"]["image"])))
            sys.exit(1)
        id_indexed[label_studios_unique_id(example["data"]["image"])] = example
    return id_indexed
"""
    Only care about image names for ids. Only works for at least labelled examples.
"""
def rect2poly(rect_path, tar_poly_id, poly_path: None):
    verify_file_exists(rect_path)
    append_mode = poly_path is not None
    
    if append_mode:
        verify_file_exists(poly_path)

    with open(rect_path) as json_file:
        rect_data = json.load(json_file)

    # for each example, change the label fields to polygon
    rect_indexed = examples_to_image_id_index(rect_data)

    if append_mode:
        print("Unimplemented")
        sys.exit(1)
    else:
        poly_indexed = convert_rect_to_poly(tar_poly_id, rect_indexed)
        poly_output = []
        for index in poly_indexed:
            poly_output.append(poly_indexed[index])
        print(json.dumps(poly_output))

    #convert from id indexed to array
    return

"""
    TODO: implement append mode
    TODO: create test for verifying append mode works
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a rectangle labeled project to a polygon labeled project')
    parser.add_argument('--rect', metavar='rect', required=True,
                        help='the json export of labelled images in a label studios project with a rectangle based labeling method.')
    parser.add_argument('--target_project_id', metavar='tar_poly_id', required=True,
                        help='The integer indexed project id found in the urls of the project page for the polygon based labeling method.')
    parser.add_argument('--poly', metavar='poly', required=False,
                        help='the json export of labelled images in a label studios project with a polygon based labeling method.')
    
    args = parser.parse_args()
    rect2poly(rect_path=args.rect, tar_poly_id=args.target_project_id, poly_path=args.poly)