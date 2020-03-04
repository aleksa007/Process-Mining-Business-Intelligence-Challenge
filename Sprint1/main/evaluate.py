import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def main():
    parser = argparse.ArgumentParser("PM_tool")
    parser.add_argument("csv_file", help="Predictions will be saved into a new CSV with this name", type=str)

    args = parser.parse_args()

    #path = os.getcwd()
    #os.chdir(path)

    csv_path = args.csv_file
    #csv_path = './output.csv'
    pre_df = pd.read_csv(csv_path, error_bad_lines=False)


    time_real = np.array(pre_df['TimeDifference'])
    time_pred = np.array(pre_df['Predicted_TimeDifference'])

    time_pred = np.nan_to_num(time_pred)
    time_real = np.nan_to_num(time_real)

    rms = np.sqrt(mean_squared_error(time_real, time_pred))
    print('Root mean squared error for time difference prediction: {}'.format(rms))

    event_real = np.array(pre_df['Event_Name'])[1:]
    event_pred = np.array(pre_df['Predicted_Event'])[:-1]

    acc = accuracy_score(event_real, event_pred)
    print('Accuracy for event prediction: {}'.format(acc))

main()

