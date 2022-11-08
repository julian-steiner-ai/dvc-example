import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open('params.yaml', encoding='UTF-8'))
prepare_stage_params = params["prepare"]

seed = params["seed"]
test_size = prepare_stage_params["test_size"]
column_names = prepare_stage_params["column_names"]

data = pd.read_csv('data/ENB2012_data.csv', sep=';')
data = data.rename(columns=column_names)

train_df, test_df = train_test_split(data, test_size=test_size, random_state=seed)

train_df.to_csv('data/train.csv', index=False, sep=';')
test_df.to_csv('data/test.csv', index=False, sep=';')