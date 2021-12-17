
from glob import glob
from os import path
import json, operator

model_dir="checkpoints/custom/tf"

results = {}
results_1c = {}
cpp = {}

def format_paper_name(run_name):
    paper_name = run_name.split("/")[3] \
        .replace("vanilla-101_", "r101_") \
        .replace("vanilla-101-tf", "r101-tf") \
        .replace("vanilla-101-", "x101-") \
        .replace("vanilla_1", "r50") \
        .replace("bert_384_", "") \
        .replace("all-MiniLM-L6-v2", "MiniLM") \
        .replace("all-mpnet-base-v2", "all-mpnet") \
        .replace("multi-qa-mpnet-base-dot-v1", "mq-mpnet") \
        .replace("all-distilroberta-v1", "distilroberta") \
        .replace("KBLab/sentence-bert-swedish-cased", "KBLab") \
        .replace("KB/bert-base-swedish-cased", "KB") \
        .replace("_1", "") \
        .replace("-tf", "")
    #return paper_name
    if not paper_name.startswith("r50") and not paper_name.startswith("x101"):
        if "-101-64x4d" in paper_name:
            return "x101-64x4d-" + paper_name.replace("-101-64x4d", "")
        elif "-101-32x4d" in paper_name:
            return "x101-32x4d-" + paper_name.replace("-101-32x4d", "")
        elif "-101-32x8d" in paper_name:
            return "x101-32x8d-" + paper_name.replace("-101-32x8d", "")
        elif "-101" in paper_name:
            return "r101-" + paper_name.replace("-101", "")
        elif not "101" in paper_name:
            return "r50-" + paper_name
    return paper_name

def compare():
    dirs = glob(path.join(model_dir, "*/"))
    for dir in dirs:
        if "ratio" in dir:
            continue

        fnames = glob(dir + "eval*.json")
        if len(fnames) == 0:
            print("no eval file found on path {}".format(dir))
            continue
        fname = fnames[-1]
        if path.exists(fname):
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) > 0:
                    last_line = lines[-1]
                    d = json.loads(last_line)
                    d = d['metric']
                    if not "0_bbox_mAP" in d.keys():
                        continue

                    del d['0_segm_mAP_copypaste']
                    del d['1_segm_mAP_copypaste']
                    del d['2_segm_mAP_copypaste']
                    del d['0_bbox_mAP_copypaste']
                    del d['1_bbox_mAP_copypaste']
                    del d['2_bbox_mAP_copypaste']
                    paper_name = format_paper_name(fname)
                    for key in d.keys():
                        f_key = key.replace('0_','in_', 1) if key.startswith('0_') else key
                        f_key = key.replace('1_','near_', 1) if key.startswith('1_') else f_key
                        f_key = key.replace('2_','out_', 1) if key.startswith('2_') else f_key
                        if not paper_name.endswith("-1c"):
                            if f_key not in results:
                                results[f_key] = {}
                            results[f_key][paper_name] = d[key]
                        else:
                            if f_key not in results_1c:
                                results_1c[f_key] = {}
                            results_1c[f_key][paper_name] = d[key]

                else:
                    print("Broken file: {}".format(dir))
        else:
            print("{} has not latest.log.val.json".format(dir))
    print(results)
    superkeys = ["in_segm_mAP", "near_segm_mAP", "out_segm_mAP"]
    for skey in superkeys:
        print("{} \t   {}       {}    {}    {}      {}   {}".format(skey, "m", "m_50", "m_75", "bb", "bb_50", "bb_75").expandtabs(30))
        sort_results = dict(sorted(results[skey].items(), key=lambda item: item[1], reverse=True))
        for key in sort_results.keys():
            print(("{} \t & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \hline".format(key, sort_results[key], results[skey + "_50"][key], results[skey + "_75"][key], results[skey.replace('segm', 'bbox')][key], results[skey.replace('segm', 'bbox') + "_50"][key], results[skey.replace('segm', 'bbox') + "_75"][key])).expandtabs(30))
        print()
    print("--------------------------- 1 class ------------------------------")
    for skey in superkeys:
        print("{} \t   {}       {}    {}    {}      {}   {}".format(skey, "m", "m_50", "m_75", "bb", "bb_50", "bb_75").expandtabs(30))
        sort_results = dict(sorted(results_1c[skey].items(), key=lambda item: item[1], reverse=True))
        for key in sort_results.keys():
            print(("{} \t & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \hline".format(key, sort_results[key], results_1c[skey + "_50"][key], results_1c[skey + "_75"][key], results_1c[skey.replace('segm', 'bbox')][key], results_1c[skey.replace('segm', 'bbox') + "_50"][key], results_1c[skey.replace('segm', 'bbox') + "_75"][key])).expandtabs(30))
        print()


if __name__ == "__main__":
    compare()