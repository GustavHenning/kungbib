from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os, json

DATASET="/data/gustav/datalab_data/poly-dn-2010-2020-720"

config_file = 'configs/gustav/kungbib-cascade-mask.py'
checkpoint_file = 'checkpoints/custom/tf/vanilla/latest.pth'
device = 'cuda:0'


# init a detector
model = init_detector(config_file, checkpoint_file, device=device)

with open(DATASET + "/valid_annotations.json", encoding='utf-8') as fh:
    data = json.load(fh)
    for d in data["images"]:
        img = DATASET + "/" + d["file_name"]
        results = inference_detector(model, img)
        show_result_pyplot(model, img, results, score_thr=0.6)
