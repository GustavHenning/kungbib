
from glob import glob
from os import path
import json, operator

model_dir="checkpoints/custom/tf"

results = {}

def compare():
    dirs = glob(path.join(model_dir, "*/"))
    for dir in dirs:
        latest = path.join(dir, "latest.log.val.json")
        if path.exists(latest):
            with open(latest, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) > 0:
                    last_line = lines[-1]
                    d = json.loads(last_line)
                    if not "0_bbox_mAP" in d.keys():
                        continue
                    for key in d.keys():
                        if not key in results.keys():
                            results[key] = {}
                        results[key][dir] = d[key]
                else:
                    print("Broken file: {}".format(dir))
        else:
            print("{} has not latest.log.val.json".format(dir))
    for key in results.keys():
        print(key)
        sorted_x = sorted(results[key].items(), key=operator.itemgetter(1), reverse=True)
        for k, value in sorted_x:
            print("{}: {}".format(value, k))


if __name__ == "__main__":
    compare()