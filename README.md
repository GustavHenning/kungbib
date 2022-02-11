# kungbib

Repository for the master's thesis "News article segmentation using multimodal input: Using Mask R-CNN and sentence transformers"

**Developer note: All scripts and programs found in this repo may not be repeatable. Read through the code before executing it to avoid unpleasant surprises.**

Since the pipeline for data may differ from your implementation, this repo should serve as inspiration to your own solution. 
It demonstrates well what parts of the MMDetection framework needs to be extended to achieve a similar result.

### MMDetection

The `mmdetection` subfolder contains scripts and programs to run training and inference of different models. The file `symlinks.txt` demonstrates what symbolic links are used.

The folder `mm_additions` contains files that are altered in the MMDetection framework. It can be summarized to contain:

* Alterations to files specifically for this dataset and domain, such as alterations of classes.

* Alterations of files that assume an image has 3 channels, often related to visualization of data.

* Additions of preprocessing steps such that the textual embedding map can be created and inserted before training and inference.

#### Scripts using runtime configuration

The `mmdetection` subfolder contains a number of scripts to alternate settings and model parameters at runtime. Once compiled and training has started, 
the full configuration can be found in the designated `checkpoints` folder for that run/model. When debugging parameterization, this file can be inspected
to ensure that the parameters are received correctly by the framework.

### Rect2poly

The `rect2poly` subfolder contains code related to the conversion of rectangle based annotations to polygon based annotations.

### Train, test, split

The `traintestsplit` subfolder contains code related to splitting datasets.


