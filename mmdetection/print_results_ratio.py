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
            dirname = RATIO_EXPERIMENT_ROOT_FOLDER + "run_{}/vanilla_1_{}{}/".format(run, ratio, cfg)
            fnames = glob(dirname + "eval*.json")
            if len(fnames) == 0:
                print("no eval file found on path {}".format(dirname))
                sys.exit()
            fname = fnames[-1]
            if not os.path.isfile(fname):
                print(fname + " could not be found!")
                sys.exit()
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                data = eval(last_line)
                data = data['metric']
                del data['0_segm_mAP_copypaste']
                del data['1_segm_mAP_copypaste']
                del data['2_segm_mAP_copypaste']
                del data['0_bbox_mAP_copypaste']
                del data['1_bbox_mAP_copypaste']
                del data['2_bbox_mAP_copypaste']
                for key in data:
                    #print(key)
                    ratio_key = "{}_{}".format(key, ratio)
                    if ratio_key.startswith('0_'):
                        ratio_key = ratio_key.replace('0_','in_', 1)
                    if ratio_key.startswith('1_'):
                        ratio_key = ratio_key.replace('1_', 'near_', 1)
                    if ratio_key.startswith('2_'):
                        ratio_key = ratio_key.replace('2_', 'out_', 1)

                    if ratio_key not in stats:
                        stats[ratio_key] = []
                    stats[ratio_key].append(data[key])
    #print(stats)
    sets = ["in", "near", "out"]
    for set in sets:
        print(set)
        for ratio in ratios:
            avg_stdev = statistics.mean([statistics.stdev(stats["{}_segm_mAP_{}".format(set, ratio)]) ,
                            statistics.stdev(stats["{}_segm_mAP_50_{}".format(set, ratio)]),
                            statistics.stdev(stats["{}_segm_mAP_75_{}".format(set, ratio)]),
                            statistics.stdev(stats["{}_bbox_mAP_{}".format(set, ratio)]),
                            statistics.stdev(stats["{}_bbox_mAP_50_{}".format(set, ratio)]),
                            statistics.stdev(stats["{}_bbox_mAP_75_{}".format(set, ratio)])])
            print("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}"
                    .format(
                        "{:.0f}\%".format(ratio*100),
                        statistics.mean(stats["{}_segm_mAP_{}".format(set, ratio)]), 
                        statistics.mean(stats["{}_segm_mAP_50_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_segm_mAP_75_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_bbox_mAP_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_bbox_mAP_50_{}".format(set, ratio)]),
                        statistics.mean(stats["{}_bbox_mAP_75_{}".format(set, ratio)]),
                        avg_stdev))
for cfg in class_configs:
    print(cfg)
    generate_averages(cfg)
