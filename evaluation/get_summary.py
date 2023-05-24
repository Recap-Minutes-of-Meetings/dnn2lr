
import os
import pickle
import pandas as pd
from tqdm import tqdm
import joblib

paths = ["../models/linear/output/",
         "../models/lgb/output/",
         "../models/nn/output/"]

log_df = pd.DataFrame()
i = 0

for path in paths:
    if not os.path.exists(path):
        continue
    
    configs = os.listdir(path)

    for conf in tqdm(configs):
        if not os.path.exists(path + conf + "/log.pickle"):
            continue
        
        log = joblib.load(path + conf + "/log.pickle")
        
        try:
            for k, v in log.items():
                try:
                    log_df.loc[i, k] = v
                except:
                    try:
                        log_df.loc[i, k] = str(v)
                    except:
                        pass
            
            log_df.loc[i, "log_path"] = os.path.abspath(path + conf + "/log.pickle")
            i += 1
        except:
            pass

log_df.to_csv('../output/run_summary.csv', index=False)
