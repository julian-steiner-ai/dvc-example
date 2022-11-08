import yaml
import pandas as pd

from joblib import dump
from catboost import CatBoostRegressor

clf = CatBoostRegressor()

params = yaml.safe_load(open('params.yaml', encoding='UTF-8'))
train_and_evaluate_params = params['train_and_evaluate']

train_df = pd.read_csv("data/train.csv", sep=';')

X_train = train_df[train_and_evaluate_params['x_columns']]
y_train = train_df[train_and_evaluate_params['y_column']]

clf.fit(X_train, y_train)

dump(clf, "models/catboost.joblib")