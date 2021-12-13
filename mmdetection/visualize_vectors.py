from glob import glob
import json, sys, pandas as pd
from random import normalvariate
import os.path
import statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

WEIGHTS="/data/gustav/datalab_data/model/vectors/all-mpnet-base-v2/"

with open(WEIGHTS + "cache.json", 'r') as f:
    data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    print(df.std(numeric_only=True).min())
    print(df.std(numeric_only=True).max())
    normalized_df=(df-df.mean())/df.std()
    print(normalized_df)
