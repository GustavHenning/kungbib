from glob import glob
import json, sys
import os.path
import statistics

RATIO_EXPERIMENT_ROOT_FOLDER="checkpoints/custom/tf/ratio/"

runs = 5
ratios = [0.25, 0.5, 0.75, 0.9, 1.0]
class_configs = ["", "-1c"]

def generate_averages(cfg):
    stats = {}

    for ratio in ratios:
        for run in range(1, runs+1):
            fname = RATIO_EXPERIMENT_ROOT_FOLDER + "run_{}/vanilla_1_{}{}/*replace_me_with_eval_stats_TODO_replace_0_with_in_etc*.json".format(run, ratio, cfg)
            if not os.path.isfile(fname):
                print(fname + " could not be found!")
                sys.exit()
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                data = eval(last_line)
                del data['in_segm_mAP_copypaste']
                del data['near_segm_mAP_copypaste']
                del data['out_segm_mAP_copypaste']
                del data['in_bbox_mAP_copypaste']
                del data['near_bbox_mAP_copypaste']
                del data['out_bbox_mAP_copypaste']
                for key in data:
                    ratio_key = "{}_{}".format(key, ratio)
                    if ratio_key not in stats:
                        stats[ratio_key] = []
                    stats[ratio_key].append(data[key])
    print(stats)
    sets = ["in", "near", "out"]
    for set in sets:
        print(set)
        for ratio in ratios:
            print("{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}"
                    .format(
                        statistics.mean(stats["{}_segm_mAP_{}".format(set, ratio)]), 
                        statistics.mean(stats["{}_segm_mAP_50_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_segm_mAP_75_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_bbox_mAP_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_bbox_mAP_50_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_bbox_mAP_75_{}".format(set, ratio)])))

for cfg in class_configs:
    generate_averages(cfg)
