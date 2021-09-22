Could not commit files in these directories because git doesnt know about symlinks.

I've copied the folder structure for where these files are found.

The following folders have been symlinked:
checkpoints, pipelines, configs, tools

Once copied to the mmdetection equivalent folders, training is done by running: 

python -W ignore tools/train.py configs/gustav/kungbib-cascade-mask-tf.py --work-dir=checkpoints/custom/tf