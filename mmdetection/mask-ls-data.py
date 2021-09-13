from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = 'configs/gustav/kungbib-cascade-mask.py'

checkpoint_file = 'checkpoints/resnet50_caffe-788b5fa3.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = '/data/gustav/datalab_data/images/dark-9771302_part02_page032.jpg'
results = inference_detector(model, img)

show_result_pyplot(model, img, results, score_thr=0.6)