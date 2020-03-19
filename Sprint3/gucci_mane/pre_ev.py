import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

'''For accuracy and rmse - do it for all three new columns'''
def pre_road(train_path: str, test_path: str) -> object:
    df2_test = pd.read_csv(train_path)
    df2_train = pd.read_csv(test_path)

    # Convert from string to datetime
    df2_train['event time:timestamp'] = pd.to_datetime(df2_train['event time:timestamp'])
    df2_test['event time:timestamp'] = pd.to_datetime(df2_test['event time:timestamp'])

    # Concatenate the datasets
    df2 = pd.concat([df2_train, df2_test])

    # Sort the values and reset the index accordingly
    df2 = df2.sort_values(by=['event time:timestamp'])
    df2 = df2.reset_index(drop=True)

    # Create dataframe with first events of each case in chronological order
    df_cases_first_event = df2.drop_duplicates(subset='case concept:name', keep='first')
    df_cases_first_event = df_cases_first_event.reset_index(drop=True)

    # Split the cases in 80-20
    index80 = int(len(df_cases_first_event) * 80 / 100 - 1)
    df_cases_train = df_cases_first_event.loc[:index80]
    df_cases_test = df_cases_first_event.loc[index80 + 1:]

    # Training and test set complete
    df_train_new = df2.loc[df2['case concept:name'].isin(list(df_cases_train['case concept:name']))]
    df_test_new = df2.loc[df2['case concept:name'].isin(list(df_cases_test['case concept:name']))]

    # Setting the 'now' (as the last event of the last case in the training set)
    # now = df_train_new[df_train_new['case concept:name']==df_cases_train.iloc[-1]\
    #    ['case concept:name']].iloc[-1]['event time:timestamp']

    now = pd.Timestamp('2012-12-12')

    # Training and test set before 'now'
    df_train_new_now = df_train_new.loc[(df_train_new['event time:timestamp'] <= now)]
    df_test_new_now = df_test_new.loc[(df_test_new['event time:timestamp'] <= now)]

    # Add argument, if -p, output the preprocessed files as csv's
    # df_test_new_now.to_csv('./data/road-test-pre.csv', index=False)
    # df_train_new_now.to_csv('./data/road-train-pre.csv', index=False)

    return df_train_new_now, df_test_new_now

def pre_loan():
    pass

def run_pre():
    '''Check which data is being used and call that function'''

def evaluate_acc_rmse(base, perm, d_tree):

    # Baseline Evaluation
    event_real = np.array(base['Event_Name'])
    event_real = event_real[1:]

    time_real = np.array(base['Real_TimeDiff'])

    event_pred_baseline = np.array(base['Baseline_Event'])[:-1]
    # event_pred_baseline = event_pred_baseline[:-1]
    time_pred_baseline = np.array(base['Baseline_TimeDiff'])

    acc_baseline = accuracy_score(event_real, event_pred_baseline)
    print('Accuracy for event prediction BASELINE: {}%'.format(round(acc_baseline, 2) * 100))
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

    # RMSE
    time_pred_tree = np.array(d_tree['DTree_TimeDiff'])

    rms_tree = np.sqrt(mean_squared_error(time_real, time_pred_tree))
    print('Root mean squared error for time difference prediction COMBINATIONS: {}'.format(round(rms_tree, 2)))


    return print('Done.')


if __name__ == '__main__':

    train, test = pre_road('./data/road-train.csv', './data/road-test.csv')
    train.to_csv('./build/road-train-pre.csv')
    test.to_csv('./build/road-test-pre.csv')
