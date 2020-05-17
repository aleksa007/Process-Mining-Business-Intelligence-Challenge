from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

import sys


def evaluate_acc_rmse(base, perm, d_tree):
    orig_stdout = sys.stdout
    sys.stdout = open('./build/report.txt', 'w')

    # Baseline Evaluation
    event_real = np.array(base['Event_Name'])[1:]

    time_real = np.array(base['Real_TimeDiff'])

    event_pred_baseline = np.array(base['Baseline_Event'])[:-1]
    time_pred_baseline = np.array(base['Baseline_TimeDiff'])

    acc_baseline = accuracy_score(event_real, event_pred_baseline)
    print('Accuracy for event prediction BASELINE: {}%'.format(round(acc_baseline, 2) * 100))

    if pd.Series(time_pred_baseline).isnull().values.any():
        time_pred_baseline = [i if (not np.isnan(i)) else 5. for i in time_pred_baseline]

    rms_baseline = np.sqrt(mean_squared_error(time_real, time_pred_baseline))
    print('Root mean squared error for time difference prediction BASELINE: {}'.format(round(rms_baseline, 2)))

    # Combinations Algorithm Evaluation
    event_real = np.array(perm['Current_Event'])
    event_real = event_real[1:]

    event_pred_combs = np.array(perm['Combs_Event'])[:-1]
    time_pred_combs = np.array(perm['Combs_TimeDiff'])

    acc_combs = accuracy_score(event_real, event_pred_combs)
    print('Accuracy for event prediction COMBINATIONS: {}%'.format(round(acc_combs, 2) * 100))

    rms_combs = np.sqrt(mean_squared_error(time_real, time_pred_combs))
    print('Root mean squared error for time difference prediction COMBINATIONS: {}'.format(round(rms_combs, 2)))

    # Decision Tree Evaluation
    event_real = np.array(d_tree['Next_Event'])  # taking next event col. as an array
    event_pred = np.array(d_tree['DTree_Event'])  # taking the predictions as an array

    acc_tree = accuracy_score(event_real, event_pred)  # calculates the accuracy based on the both arrays
    print('Accuracy for event prediction DECISION TREE: {}%'.format(round(acc_tree, 2) * 100))

    # RMSE D Tree
    time_real_tree = np.array(d_tree['DTree_Actual_TimeDiff'])
    time_pred_tree = np.array(d_tree['DTree_TimeDiff'])

    rms_tree = sqrt(mean_squared_error(time_real_tree, time_pred_tree))/60/60/24
    print('Root mean squared error for time difference prediction DECISION TREE: {}'.format(round(rms_tree, 2)))

    sys.stdout.close()
    sys.stdout = orig_stdout

    return None
