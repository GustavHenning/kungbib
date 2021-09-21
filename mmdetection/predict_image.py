from mmdet.apis import init_detector, inference_detector, show_result_pyplot

device = 'cuda:0'

config_file = 'configs/gustav/kungbib-cascade-mask.py'
checkpoint_file = 'checkpoints/custom/latest.pth'

img_to_predict = '/data/gustav/datalab_data/images/dark-4411342_part01_page002.jpg'

# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
results = inference_detector(model, img_to_predict)

show_result_pyplot(model, img_to_predict, results, score_thr=0.6)