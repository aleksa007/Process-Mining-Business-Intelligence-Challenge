import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def main():
    parser = argparse.ArgumentParser("PM_tool")
    parser.add_argument("csv_file", help="Predictions will be saved into a new CSV with this name", type=str)

    args = parser.parse_args()

    csv_path = args.csv_file

    ###########
    predicted_df = pd.read_csv(csv_path, error_bad_lines=False)


    event_real = np.array(predicted_df['Current_Event'])
    event_pred = np.array(predicted_df['Predicted_Event'])
    event_real = event_real[1:]
    event_pred = event_pred[:-1]

    acc = accuracy_score(event_real, event_pred)
    print('Accuracy for event prediction TEST SET: {}%'.format(round(acc, 2) * 100))

    time_real = np.array(predicted_df['Real_Diff'])
    time_pred = np.array(predicted_df['Predicted_Diff'])


    rms = np.sqrt(mean_squared_error(time_real, time_pred))
    print('Root mean squared error for time difference prediction TEST SET: {}'.format(round(rms, 2)))
    ##########

main()

