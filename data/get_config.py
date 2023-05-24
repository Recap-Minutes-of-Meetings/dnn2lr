import os
import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import pathlib

def preprocces_train(data_name, train, roles, n_samples=10_000, **kwargs):
    if data_name == 'bnp':
        pass
    elif data_name == 'credit':
        pass
    
    train_len = len(train)
    tr_indx, _, _, _ = train_test_split(
            np.arange(train_len), np.arange(train_len), train_size=n_samples, random_state=42, stratify=train[roles["target"]].values)
    
    train = train.iloc[tr_indx.tolist(), :]
    train.fillna(train.median(), inplace=True)
    for col in roles['category']:
        le = preprocessing.LabelEncoder()
        le.fit(train[col].values)
        train[col] = (le.transform(train[col]) + 1).astype(int).astype(str)
    return train

def get_roles(data_name):
    if data_name == 'bnp':
        return {'category': ['v31', 'v74', 'v3', 'v66', 'v110', 'v75', 'v24', 'v91', 'v30', 'v107', 'v62', 'v71', 'v129', 'v47', 'v38', 'v52', 'v72', 'v79', 'v112', 'v113', 'v125', 'v56'],
                'drop': ["ID", "v22"],
                'target': "target",
                'task': "binary"
                }
    elif data_name == 'credit':
        return {'category': ['Month', 'Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount'],
                'drop': ["ID", "Customer_ID", "Name", "Type_of_Loan"],
                'target': "Credit_Score",
                'task': "multiclass",
                }
    else:
        raise NotImplementedError()


cols = ["name", "train_path"]
info = [
    ["credit", "credit.csv"],
    ["bnp", "bnp.csv"],
]

dt = pd.DataFrame(info, columns=cols)

for i, r in dt.iterrows():
    train = pd.read_csv(r["train_path"])
    nrow, ncol = train.shape
    dt.loc[i, "nrow"] = int(nrow)
    dt.loc[i, "ncol"] = int(ncol)

dt.to_csv("./dataset_info.csv", index=False)
data_info = pd.read_csv("./dataset_info.csv")

logs = []

for i, r in tqdm(data_info.iterrows(), total=len(data_info)):
    train = pd.read_csv(r["train_path"])
    roles = get_roles(r['name'])
    train = preprocces_train(r['name'], train, roles)
    
    y = train[roles["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        y, y, test_size=0.3, random_state=42, stratify=y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        y_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    paths = {}
    for ind, name in zip([X_train.index, X_valid.index, X_test.index],
                         ['train', 'valid', 'test']):
        dt = train.loc[ind, :]
        path = "./splits/" + r['name']
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path += "/" + name + '.csv'
        dt.to_csv(path, index=False)
        paths[name + "_path"] = os.path.abspath(path)
    
    log = {
        **roles,
        **paths
    }
    logs.append(log)

joblib.dump(logs, "./data_configs.pkl")
