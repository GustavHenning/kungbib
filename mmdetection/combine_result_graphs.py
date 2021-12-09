import numpy as np
import PIL, os
from os import path
from PIL import Image
from glob import glob

model_dir="./checkpoints/custom/tf/"

def stack(imgs):
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    return np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) ) 

def stack_images(path):
    allareas = glob(path + "/*allarea.png")
    large = glob(path + "/*large.png")
    medium = glob(path + "/*medium.png")
    small = glob(path + "/*small.png")
    bar = glob(path + "/*bar plot.png")

    imgs_all    = [ PIL.Image.open(i) for i in sorted(allareas) ]
    imgs_large    = [ PIL.Image.open(i) for i in sorted(large) ]
    imgs_medium    = [ PIL.Image.open(i) for i in sorted(medium) ]
    imgs_small    = [ PIL.Image.open(i) for i in sorted(small) ]
    imgs_bar    = [ PIL.Image.open(i) for i in sorted(bar) ]

    stacked_all = stack(imgs_all)
    stacked_large = stack(imgs_large)
    stacked_medium = stack(imgs_medium)
    stacked_small = stack(imgs_small)
    stacked_bar = stack(imgs_bar)

    imgs_comb = np.hstack([stacked_all, stacked_large, stacked_medium, stacked_small, stacked_bar])

    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    model_name = path.split("/")[-3]
    imgs_comb.save( path + "-" + model_name + '-stacked.png' )    

    # for a vertical stacking it is simple: use vstack
    #imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    #imgs_comb = PIL.Image.fromarray( imgs_comb)
    #imgs_comb.save( 'stacked_v.jpg' )

def combine_results(dir):

    test_sets = glob(path.join(dir, "results", "*/"))
    for set in test_sets:
        bbox = path.join(set, "bbox")
        if path.exists(bbox):
            stack_images(bbox)
        segm = path.join(set, "segm")
        if path.exists(segm):
            stack_images(segm)

if __name__ == "__main__":
    runs = 5
    for i in range(1,runs+1):
        dirs = glob(path.join(model_dir, "ratio", "run_{}".format(i), "*/"))
        for dir in dirs:
            print(dir)
            combine_results(dir)


    dirs = glob(path.join(model_dir, "*/"))
    for dir in dirs:
        if "ratio" in dir:
            continue
        print(dir)
        combine_results(dir)