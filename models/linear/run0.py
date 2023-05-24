import os
import subprocess

from joblib import Parallel, delayed
import pickle
from tqdm import tqdm

configs = os.listdir("./input/")

def run(config_name):
    r = subprocess.call(['python3', "run_config.py",
                         '--config_name', config_name])
    print(r)
    print("=============")
    return 0

for conf in tqdm(configs):
    with open('./input/' + conf, 'rb') as f:
        c = pickle.load(f)
    print(c)
    print("-------------")
    run(conf)

# Parallel(n_jobs=2)(delayed(run)(conf) for conf in tqdm(configs))
