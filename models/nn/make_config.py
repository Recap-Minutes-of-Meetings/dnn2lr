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


for model in ["dense", "resnet", "denselight", "mlp"]:
    for lr in [1e-3, 3e-4, 5e-5]:
        for config in configs:
            c = deepcopy(config)
            cat_len = len(c['category'])
            
            c["model"] = model
            c["opt_params"] = { 'lr': lr, 'weight_decay': 0 }
            c["gpu"] = str(np.random.choice(["1"]))
            c["cross_f"] = None
                
            exp_name = str(c)
            random.seed(exp_name)
            c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
            joblib.dump(c, path + c["exp_id"] + ".pkl")

# 12174