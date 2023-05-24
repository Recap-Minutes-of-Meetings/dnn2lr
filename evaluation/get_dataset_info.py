import joblib
import pandas as pd
from tqdm import tqdm
from pprint import pprint

data_path = "../data/data_configs.pkl"
configs = joblib.load(data_path)

data_df = pd.DataFrame(columns=["data_name", "nrow", "ncol", "ncat", "nnum"])
for i, (log) in tqdm(enumerate(configs), total=len(configs)):
    train = pd.read_csv(log["train_path"])
    test = pd.read_csv(log["test_path"])
    
    nrows = train.shape[0]
    ncols = train.shape[1] - 1 - len(log["drop"])
    ncat = len(log["category"])
    nrel = ncols - ncat
    data_df.loc[i] = [log['train_path'].split("/")[-2], nrows, ncols, ncat, nrel]

data_df.sort_values("nrow").to_csv("../output/dataset_info.csv", index=False)
