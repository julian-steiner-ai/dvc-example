import os
import sys
import json
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

if not os.path.exists('plots'):
    os.mkdir('plots')

test_df = pd.read_csv('data/')