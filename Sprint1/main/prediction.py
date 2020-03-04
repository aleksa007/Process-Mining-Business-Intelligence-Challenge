import matplotlib.pyplot as plt;
import pandas as pd

plt.rcdefaults()
import numpy as np
import os
import datetime
from progressbar import ProgressBar
import math
import time
import argparse


def main():
    parser = argparse.ArgumentParser("PM_tool")
    parser.add_argument("train_file", help="Process mining model will be trained on this", type=str)
    parser.add_argument("test_file", help="To test the predictions on, and evaluate later on.", type=str)
    parser.add_argument("output_file", help="Predictions will be saved into a new CSV with this name", type=str)

    args = parser.parse_args()

    train_path = args.train_file
    test_path = args.test_file
    out_path = args.output_file

    start_time = time.time()

    # train_path = './data/road_train.csv'
    # test_path = './data/road_test.csv'

    #print(train_path, test_path)

    data_train = pd.read_csv(train_path, error_bad_lines=False, parse_dates=[4], index_col = False)
    data_test = pd.read_csv(test_path, error_bad_lines=False, parse_dates=[4], index_col = False)

    # Keeping only the date of the timestamp and removing the hours.

    if train_path == './data/road-train-pre.csv':
        (eventID, caseID, event_name, transition, stamp) = ('eventID','case_concept_name','event_concept_name','event_lifecycle_transition','event_time_timestamp')
    else:
        (eventID, caseID, event_name, transition, stamp) = ("eventID ","case concept:name","event concept:name","event lifecycle:transition","event time:timestamp")

    data_train[stamp] = pd.to_datetime(data_train[stamp])
    data_test[stamp] = pd.to_datetime(data_test[stamp])

    # Sorting the frame chronologically and per case.
    data_train = data_train.sort_values(by=[caseID, stamp])
    data_test = data_test.sort_values(by=[caseID, stamp])

    # New .csv file with the fixed sorting and timestamp
    data_train.to_csv("fixed_train.csv")
    data_test.to_csv("fixed_test.csv")
    # Creating a dictionary file where the case names (IDs) are the keys and the values are a list of two sublists.
    # The first sublist contains all event names of a case, sorted chronologically.
    # The second sublist contains all event time stamps of a case, sorted chronologically.

    # file = open('fixed.csv', 'r')
    log = dict()
    #print('making dictionary from fixed file')
    with open('fixed_train.csv', 'r') as f:
        next(f)

        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[2]
            task = parts[3]
            timestamp = parts[5]

            if caseid not in log:
                log[caseid] = [[], []]

            log[caseid][0].append(task)
            log[caseid][1].append(timestamp)

    f.close()

    log_test = dict()
    #print('making test dictionary from fixed file')
    with open('fixed_test.csv', 'r') as f:
        next(f)

        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[2]
            task = parts[3]
            timestamp = parts[5]

            if caseid not in log_test:
                log_test[caseid] = [[], []]

            log_test[caseid][0].append(task)
            log_test[caseid][1].append(timestamp)

    f.close()

    #print('done making dictionaries')
    #print('deleting fixed files')
    os.remove('fixed_train.csv')
    os.remove('fixed_test.csv')

    #print('done')

    ### Event prediction

    event_list = []  # list of lists that contains all event names for every case, where each sublist is a new case.
    for case in log.keys():
        event_list.append((log[case][0]))

    events_longest_event = max(map(len, event_list))  # longest event

    pos_events = []  # list of lists which has all event names for every position, where each sublist is a new position
    for i in range(events_longest_event):  # creates a list every iteration which is appended to the above list
        pos_events.append([])
        for case in event_list:
            if (len(case) - 1) >= i:
                pos_events[i].append(case[i])
            else:
                pass

    cases = list(data_train[event_name].unique())  # list of all unique event names
    event_frequency = []
    index = 0
    str_best = cases[0]
    for i in range(len(pos_events)):  # counts how many times each event name occurs in every position
        for x in cases:
            best = pos_events[i].count(cases[0])
            current = pos_events[i].count(x)
            if current > best:
                best = current
                str_best = x
        event_frequency.append(str_best)  # only the variable with highest count is appended to the frequency list
        index += 1
    print('Event prediction done')
    ### Timestamp prediction

    # list of lists that contains all timestamps for all events of a case, where every sublist is a new case
    time_list = []
    for case in log.keys():
        time_list.append((log[case][1]))

    times_longest_event = max(map(len, time_list))  # longest event
    #print("Longest event in train: ", times_longest_event)

    time_list = []
    for case in log_test.keys():
        time_list.append((log_test[case][1]))

    times_longest_event_test = max(map(len, time_list))  # longest event
    #print("Longest event in test: ", times_longest_event_test)

    def timeDiff(tup):
        """The function calculates the difference between two timestamps and returns the absolute value."""

        datetimeFormat = '%Y-%m-%d'
        diff = datetime.datetime.strptime(tup[0], datetimeFormat) \
               - datetime.datetime.strptime(tup[1], datetimeFormat)

        return abs(diff.days)

    pbar = ProgressBar()

    # list of lists that contains all event timestamps for every position, where each sublist is a new position
    pos_times = []

    for i in pbar(range(times_longest_event)):  # new list is created each iteration
        pos_times.append([])
        for case in time_list:  # for every case a tuple created which contains the current and the next event time
            if (len(case) - 1) > i:
                event_tpl = (case[i], case[i + 1])
                pos_times[i].append(timeDiff(event_tpl))  # the difference is calculated and saved in a list
            else:
                pass

    time_frequency = []  # list that contains all predictions for every position
    for i in pos_times:
        avg_pos = np.mean(i)  # the mean for every position is calculated and appended to the corresponding index
        if math.isnan(avg_pos) == False:  # a small check to avoid bugs because the last value is a NaN
            avg_pos = int(np.mean(i))
            time_frequency.append(avg_pos)
        else:
            time_frequency.append(avg_pos)
    print('Time prediction done')

    ### Adding predictions to the file
    # * To the test file
    for case in log_test.keys():
        n_events = len(log_test[case][0])
        n_times = len(log_test[case][1])

        if len(log_test[case][0]) > times_longest_event:
            n_unknown = len(log_test[case][0]) - times_longest_event
            evs = event_frequency[:n_events]
            evs.extend(['Unknown' for one in range(0,n_unknown)])

            tms = time_frequency[:n_times]
            tms.extend([31 for one in range(0,n_unknown)])

            log_test[case].extend((evs, tms))
        else:
            log_test[case].extend((event_frequency[:n_events], time_frequency[:n_times]))

    for case in log_test.keys():
        real_diff = []  # empty list for real time differences for each case
        for event, i in zip(log_test[case][1], range(len(log_test[case][1]))):  # add for each event the difference between this
            if (len(log_test[case][1]) - 1) > i:
                this = event
                following = log_test[case][1][i + 1]
                real_time_tup = (this, following)
                real_diff.append(timeDiff(real_time_tup))
            else:
                real_diff.append(0)
        log_test[case].extend([real_diff])

    #print(list(log_test.items())[:4])

    #[('A28905', [['Create Fine', 'Send Fine', 'Insert Fine Notification', 'Add penalty', 'Send for Credit Collection'],
    # ['2009-09-25', '2010-01-19', '2010-08-02', '2010-09-04', '2012-03-26'],
    # ['Create Fine', 'Create Fine', 'Insert Fine Notification', 'Add penalty', 'Payment'],
    # [88, 58, 68, 453, 224],
    # [116, 195, 33, 569, 0]])

    case_names = []
    event_names = []
    timestamp = []
    p_event = []
    p_timestamp = []
    time_diff = []

    for i in log_test.keys():
        for x in range(len(log_test[i][0])):
            case_names.append(i)
                # it = iter(log_test[i])
                # the_len = len(next(it))
                # if not all(len(l) == the_len for l in it):
                #     print(log_test[i])
                #     raise ValueError('not all lists have same length!')
            event_names.append(log_test[i][0][x])
            timestamp.append(log_test[i][1][x])
            p_event.append(log_test[i][2][x])
            p_timestamp.append(log_test[i][3][x])
            time_diff.append(log_test[i][4][x])

    frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
                  'TimeStamp': timestamp, 'TimeDifference': time_diff, 'Predicted_Event': p_event,
                  'Predicted_TimeDifference': p_timestamp}

    predicted_df = pd.DataFrame.from_dict(frame_dict)

    # def output_csv():

    print(predicted_df.head())

    print("Prediction Time --- %s seconds ---" % (time.time() - start_time))

    predicted_df.to_csv(out_path)

main()