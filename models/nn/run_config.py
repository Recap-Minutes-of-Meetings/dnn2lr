import os
import pandas as pd
import pathlib
import argparse
from time import time
import numpy as np
import sys
sys.path.insert(0, '../')

import joblib

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML


def main(args):
    path = str(pathlib.Path(__file__).parent.absolute()) + "/"
    log = joblib.load(path + "/input/" + args.config_name)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = log["gpu"]
    
    train = pd.read_csv(log["train_path"])
    valid = pd.read_csv(log["valid_path"])
    test = pd.read_csv(log["test_path"])

    roles = {
        'category': log["category"],
        'drop': log["drop"],
        'target': log["target"]
    }
    
    if log["cross_f"] is not None:
        for cr_f in log["cross_f"]:
            _cr_f = np.array(log['category'])[cr_f]
            new_col = ''.join([x + "_" for x in _cr_f])
            train[new_col] = (train[_cr_f.tolist()].astype(str) + '_').sum(axis=1)
            valid[new_col] = (valid[_cr_f.tolist()].astype(str) + '_').sum(axis=1)
            test[new_col] = (test[_cr_f.tolist()].astype(str) + '_').sum(axis=1)
            roles['category'].append(new_col)
    
    task = Task(log["task"], metric='crossentropy' if log['task'] != 'binary' else 'logloss')
    
    cur_path = path + '/output/' + log['exp_id'] + "/"
    pathlib.Path(cur_path).mkdir(parents=True, exist_ok=True)
    
    automl = TabularAutoML(
        task = task, 
        timeout = 99999999,
        cpu_limit = 4,
        general_params = {"use_algos": [[log["model"]]]},
        reader_params = {'n_jobs': 4, 'cv': 4, 'random_state': 0, "advanced_roles": False},
        nn_params={"grad_save_path": cur_path, "n_epochs": 100, "path_to_save": None, "bs": 1024, 
                   "emb_ratio": 1, "max_emb_size": 10, "snap_params": { 'k': 3, 'early_stopping': True, 'patience': 10, 'swa': True},
                   "opt_params": log["opt_params"]},
        nn_pipeline_params = {"use_qnt": False},
    )
    
    log["train time (sec)"] = time()
    oof_preds = automl.fit_predict(train, roles, valid_data=valid, verbose=4).data
    log["train time (sec)"] = time() - log["train time (sec)"]
    preds = automl.predict(test).data

    joblib.dump([valid[roles['target']].values, oof_preds,
                 test[roles['target']].values, preds], cur_path + 'oof_test_preds.pickle')
    joblib.dump(log, cur_path + 'log.pickle')
    
    # need to reshuffle columns in origin order
    sindx = np.argsort(roles['category'])
    for stage in ['train', 'val', 'test']:
        grads = joblib.load(cur_path + "/" + stage + "_grads.pkl")
        
        emb_dims = [x.shape[1] for x in grads['emb']]
        csum = [0] + np.cumsum(emb_dims).tolist()
        
        grads["emb"] = [grads["emb"][x] for x in sindx]
        grads["vals"] = grads["vals"][:, sindx]
        
        _grads = []
        for c_ind in range(grads['vals'].shape[1]):
            _grads.append(grads['grads'][:, csum[c_ind]:csum[c_ind + 1]])
        
        _grads = [_grads[x] for x in sindx]
        _grads = np.hstack(_grads)
        grads["grads"] = _grads
        joblib.dump(grads, cur_path + "/" + stage + "_grads.pkl")
    
    cur_path = path + '/input/' + log['exp_id'] + ".pkl"
    os.remove(cur_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params info you can get on")
    parser.add_argument('--config_name', nargs='?',
                    default='./',
                    help='')
    args = parser.parse_args()
    main(args)
