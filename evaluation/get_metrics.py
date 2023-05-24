import os
import joblib
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm


paths = ["../models/linear/output/",
         "../models/lgb/output/",
         "../models/nn/output/"]

for path in paths:
    if not os.path.exists(path):
        continue
    
    configs = os.listdir(path)

    for conf in tqdm(configs):
        
        if not os.path.exists(path + conf + "/log.pickle"):
            continue

        log = joblib.load(path + conf + "/log.pickle")
            
        y, oof, y_t, pred_t = joblib.load(path + conf + '/oof_test_preds.pickle')
            
        for stage, (true, pred) in zip(["val", "test"],
                                        [(y, oof), (y_t, pred_t)]):
            log[f'logloss_{stage}'] = log_loss(true, pred)
            if pred.shape[1] == 1:
                log[f'auc_{stage}'] = roc_auc_score(true, pred)
            
        joblib.dump(log, path + conf + "/log.pickle")
 