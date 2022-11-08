import os
import sys
import json
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, plot_roc_curve

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py path_to_model \n")
    sys.exit(1)

model_file = sys.argv[1]
clf = load(model_file)

needed_directories = ['plots', 'metrics']

for needed_directory in needed_directories:
    if not os.path.exists(needed_directory):
        os.mkdir(needed_directory)

params = yaml.safe_load(open('params.yaml', encoding='UTF-8'))
train_and_evaluate_params = params['train_and_evaluate']

test_df = pd.read_csv(os.path.join('data', 'test.csv'), sep=';')

X_test = test_df[train_and_evaluate_params['x_columns']]
y_test = test_df[train_and_evaluate_params['y_column']]

y_pred = clf.predict(X_test)

results = {
    'f1': f1_score(y_test, y_pred),
    'accuray': accuracy_score(y_test, y_pred),
}

json.dump(results, open(os.path.join('metrics', 'metrics.json'), 'w', encoding='UTF-8'))
