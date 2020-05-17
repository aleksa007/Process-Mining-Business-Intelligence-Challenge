from typing import List, Tuple, Dict, Union
import os

import pandas as pd


def load_base_dict(data_train: object, data_test: object) -> Tuple[
    Dict[Union[str, List[str]], List[list]], Dict[Union[str, List[str]], List[list]]]:
    '''

    :param data_train:
    :param data_test:
    :return:
    '''
    pt1 = data_train.columns.get_loc('case concept:name') + 1
    pt2 = data_train.columns.get_loc('event concept:name') + 1
    pt3 = data_train.columns.get_loc('event time:timestamp') + 1

    # New .csv file with the fixed sorting and timestamp
    data_train.to_csv("./build/fixed_train.csv")
    # Creating a dictionary file where the case names (IDs) are the keys and the values are a list of two sublists.
    # The first sublist contains all event names of a case, sorted chronologically.
    # The second sublist contains all event time stamps of a case, sorted chronologically.

    log = dict()
    with open('./build/fixed_train.csv', 'r') as f:
        next(f)

        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[pt1]
            task = parts[pt2]
            timestamp = parts[pt3]

            if caseid not in log:
                log[caseid] = [[], []]

            log[caseid][0].append(task)
            log[caseid][1].append(timestamp)

    f.close()

    os.remove('./build/fixed_train.csv')

    # TEST

    data_test.to_csv("./build/fixed_test.csv")

    log_test = dict()
    with open('./build/fixed_test.csv', 'r') as f:
        next(f)

        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[pt1]
            task = parts[pt2]
            timestamp = parts[pt3]

            if caseid not in log_test:
                log_test[caseid] = [[], []]

            log_test[caseid][0].append(task)
            log_test[caseid][1].append(timestamp)

    f.close()

    os.remove('./build/fixed_test.csv')

    return log, log_test


def load_comb_dict(data_train: object, data_test: object) -> object:
    pt1 = data_train.columns.get_loc('case concept:name') + 1
    pt2 = data_train.columns.get_loc('event concept:name') + 1
    pt3 = data_train.columns.get_loc('event time:timestamp') + 1
    pt4 = data_train.columns.get_loc('day_of_week') + 1

    data_train.to_csv("./build/fixed_train.csv")

    log = dict()
    with open('./build/fixed_train.csv', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')

            caseid = parts[pt1]

            task = parts[pt2]
            timestamp = parts[pt3]
            day = parts[pt4]

            if caseid not in log:
                log[caseid] = [[], [], []]

            log[caseid][0].append(task)
            log[caseid][1].append(timestamp)
            log[caseid][2].append(day)

    file.close()

    os.remove('./build/fixed_train.csv')

    # TEST

    data_test.to_csv("./build/fixed_test.csv")

    log_test = dict()

    with open('./build/fixed_test.csv', 'r') as file_test:
        next(file_test)
        for line in file_test:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')

            caseid = parts[pt1]

            task = parts[pt2]
            timestamp = parts[pt3]
            day = parts[pt4]

            if caseid not in log_test:
                log_test[caseid] = [[], [], []]

            log_test[caseid][0].append(task)
            log_test[caseid][1].append(timestamp)
            log_test[caseid][2].append(day)
    file.close()

    os.remove('./build/fixed_test.csv')

    return log, log_test


def load_tree_dict(data_train: object, data_test: object, cases: List[str]) -> Tuple[
    Dict[Union[str, List[str]], List[list]], Dict[Union[str, List[str]], List[list]]]:
    '''

    :param data_train:
    :param data_test:
    :return:
    '''
    pt1 = data_train.columns.get_loc('case concept:name') + 1
    pt2 = data_train.columns.get_loc('event concept:name') + 1
    pt3 = data_train.columns.get_loc('event time:timestamp') + 1

    data_train.to_csv("./build/fixed.csv")

    log = dict()
    with open('./build/fixed.csv', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[pt1]

            task = parts[pt2]
            # added two lines
            # first one to have numeric labels for weekdays, +len cases to prevent same labeling
            # last one is only for the predicted_df in the end
            timestamp = pd.to_datetime(parts[pt3]).weekday() + len(cases)
            timestamp_full = parts[pt3]
            if caseid not in log:
                log[caseid] = [[], [], []]

            log[caseid][0].append(task)

            # append the timestamps.
            log[caseid][1].append(timestamp_full)
            log[caseid][2].append(timestamp)

    file.close()

    os.remove('./build/fixed.csv')

    # Test

    data_test.to_csv("./build/fixed_test.csv")

    log_test = dict()
    with open('./build/fixed_test.csv', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[pt1]
            # same as training
            task = parts[pt2]
            timestamp = pd.to_datetime(parts[pt3]).weekday() + len(cases)
            timestamp_full = parts[pt3]

            if caseid not in log_test:
                log_test[caseid] = [[], [], []]

            log_test[caseid][0].append(task)
            log_test[caseid][1].append(timestamp_full)
            log_test[caseid][2].append(timestamp)

    file.close()

    os.remove('./build/fixed_test.csv')

    return log, log_test


def drop_bugs(log_test: object, data_test: object) -> object:
    """Fixing a bug of cases that are in the test data but are incomplete due to the train-test split."""

    bugs = []

    for i in log_test.keys():
        if len(log_test[i][0]) == 1:
            bugs.append(i)

    for x in bugs:
        del log_test[x]
        data_test.drop(data_test.index[data_test['case concept:name'] == x], inplace=True)

    return log_test, data_test
