import os
import joblib
from model import DNN2LR
from math import ceil 
import pathlib

base_path = "../nn/output/"
configs = []

for conf in os.listdir(base_path):
    cur_path = base_path + conf + "/"
    log = joblib.load(cur_path + "log.pickle")
    grads = joblib.load(cur_path + "train_grads.pkl")
    f_gen = DNN2LR()
    f_gen.fit(grads['grads'], grads['vals'], grads['emb'])
    
    for nu in [0.5, 0.1, 0.05, 0.01]:
        new_f = f_gen.get_cross_f(num=ceil(0.5 * grads['vals'].shape[1]), nu=nu)
        cur_conf = {"nu": nu, "cross_type": "dnn2lr", **log, "cross_f": new_f}
        cur_conf['nn_model'] = cur_conf['model']
        configs.append(cur_conf)

cur_path = "./output/"
pathlib.Path(cur_path).mkdir(parents=True, exist_ok=True)

joblib.dump(configs, cur_path + "dnn2lr_cross_f.pkl")
