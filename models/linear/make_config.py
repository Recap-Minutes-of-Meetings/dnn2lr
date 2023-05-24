from copy import deepcopy
import random
import pathlib
import joblib
import numpy as np
from itertools import combinations
from math import ceil

path = str(pathlib.Path(__file__).parent.absolute()) + "/" + "input/"
p = pathlib.Path(path)
p.mkdir(parents=True, exist_ok=True)

data_path = "../../data/data_configs.pkl"
configs = joblib.load(data_path)
dnn2lr_config = joblib.load("../dnn2lr/output/dnn2lr_cross_f.pkl")

ORDER = 3
N_NEW_CROSS_F = 0.5
N_RUNS = 30
np.random.seed(33)

def sample_comb(vals, k=3, num=10):
    if num < 1:
        num = ceil(len(vals) * num)
    
    vals = np.arange(len(vals))
    np.random.shuffle(vals)
    vals = vals[:30]
    vals = np.array(list(combinations(vals, k)))
    res = vals[np.random.choice(np.arange(len(vals)), num, replace=False)].tolist()
    return res

for config in configs:
    c = deepcopy(config)
    cat_len = len(c['category'])
    
    c["model"] = "linear"
    c["gpu"] = str(np.random.choice(["1"]))
    c["cross_f"] = None
        
    exp_name = str(c)
    random.seed(exp_name)
    c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
    joblib.dump(c, path + c["exp_id"] + ".pkl")
    
    for i in range(N_RUNS):
        c = deepcopy(config)
        cross_f = sample_comb(np.arange(cat_len), k=ORDER, num=N_NEW_CROSS_F)
        
        c["model"] = "linear"
        c["gpu"] = str(np.random.choice(["1"]))
        c["cross_f"] = cross_f
        
        exp_name = str(c)
        random.seed(exp_name)
        c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
        joblib.dump(c, path + c["exp_id"] + ".pkl")
    
    
    # dnn2lr cross_f
    for config2 in dnn2lr_config:
        c = deepcopy(config)
        c.update(config2)
        c["model"] = "linear"
        c["gpu"] = str(np.random.choice(["1"]))
        
        exp_name = str(c)
        random.seed(exp_name)
        c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
        joblib.dump(c, path + c["exp_id"] + ".pkl")
