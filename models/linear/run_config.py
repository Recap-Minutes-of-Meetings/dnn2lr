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
    
    automl = TabularAutoML(
        task = task, 
        timeout = 99999999,
        cpu_limit = 4,
        general_params = {"use_algos": [["linear_l2"]]},
        reader_params = {'n_jobs': 4, 'cv': 4, 'random_state': 0, "advanced_roles": False},
    )
    
    log["train time (sec)"] = time()
    oof_preds = automl.fit_predict(train, roles, valid_data=valid, verbose=4).data
    log["train time (sec)"] = time() - log["train time (sec)"]
    preds = automl.predict(test).data
    
    cur_path = path + '/output/' + log['exp_id'] + "/"
    pathlib.Path(cur_path).mkdir(parents=True, exist_ok=True)

    joblib.dump([valid[roles['target']].values, oof_preds,
                 test[roles['target']].values, preds], cur_path + 'oof_test_preds.pickle')
    joblib.dump(log, cur_path + 'log.pickle')
    
    cur_path = path + '/input/' + log['exp_id'] + ".pkl"
    os.remove(cur_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params info you can get on")
    parser.add_argument('--config_name', nargs='?',
                    default='./',
                    help='')
    args = parser.parse_args()
    main(args)
