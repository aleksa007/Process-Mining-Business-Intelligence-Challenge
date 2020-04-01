import datetime
import itertools
import math
import os
from itertools import tee, repeat

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


def baseline(data_train, data_test):
    (eventID, caseID, event_name, transition, stamp) = ("eventID ", "case concept:name",
                                                        "event concept:name", "event lifecycle:transition",
                                                        "event time:timestamp")
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'

    if 'case LoanGoal' in data_train.columns:
        data_train['case LoanGoal'] = data_train['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

        data_test['case LoanGoal'] = data_test['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

    elif len(data_train.columns) == 5:  # check if road data
        datetimeFormat = '%Y-%m-%d'

    pt1 = data_train.columns.get_loc('case concept:name') + 1
    pt2 = data_train.columns.get_loc('event concept:name') + 1
    pt3 = data_train.columns.get_loc('event time:timestamp') + 1

    # Sorting the frame chronologically and per case.
    data_train = data_train.sort_values(by=[caseID, stamp])
    data_test = data_test.sort_values(by=[caseID, stamp])

    # New .csv file with the fixed sorting and timestamp
    data_train.to_csv("./build/fixed_train.csv")
    data_test.to_csv("./build/fixed_test.csv")
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

    os.remove('./build/fixed_train.csv')
    os.remove('./build/fixed_test.csv')

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
    ### Timestamp prediction

    # list of lists that contains all timestamps for all events of a case, where every sublist is a new case
    time_list = []
    for case in log.keys():
        time_list.append((log[case][1]))

    times_longest_event = max(map(len, time_list))  # longest event

    time_list = []
    for case in log_test.keys():
        time_list.append((log_test[case][1]))

    times_longest_event_test = max(map(len, time_list))  # longest event

    def timeDiff(tup, datetimeFormat):
        """The function calculates the difference between two timestamps and returns the absolute value."""

        # datetimeFormat = '%Y-%m-%d'
        diff = datetime.datetime.strptime(tup[0], datetimeFormat) \
               - datetime.datetime.strptime(tup[1], datetimeFormat)

        return abs(diff.days)

    pbar = ProgressBar()

    # list of lists that contains all event timestamps for every position, where each sublist is a new position
    pos_times = []
    print('[1/18]')
    for i in pbar(range(times_longest_event)):  # new list is created each iteration
        pos_times.append([])
        for case in time_list:  # for every case a tuple created which contains the current and the next event time
            if (len(case) - 1) > i:
                event_tpl = (case[i], case[i + 1])
                pos_times[i].append(
                    timeDiff(event_tpl, datetimeFormat))  # the difference is calculated and saved in a list
            else:
                pass

    time_frequency = []  # list that contains all predictions for every position
    for i in pos_times:
        avg_pos = np.mean(i)  # the mean for every position is calculated and appended to the corresponding index
        if not math.isnan(avg_pos):  # a small check to avoid bugs because the last value is a NaN
            avg_pos = int(np.mean(i))
            time_frequency.append(avg_pos)
        else:
            time_frequency.append(avg_pos)

    ### Adding predictions to the file
    # * To the test file
    for case in log_test.keys():
        n_events = len(log_test[case][0])
        n_times = len(log_test[case][1])

        if len(log_test[case][0]) > times_longest_event:
            n_unknown = len(log_test[case][0]) - times_longest_event
            evs = event_frequency[:n_events]
            evs.extend(['Unknown' for _ in range(0, n_unknown)])

            tms = time_frequency[:n_times]
            tms.extend([31 for _ in range(0, n_unknown)])

            log_test[case].extend((evs, tms))
        else:
            log_test[case].extend((event_frequency[:n_events], time_frequency[:n_times]))

    for case in log_test.keys():
        real_diff = []  # empty list for real time differences for each case
        for event, i in zip(log_test[case][1],
                            range(len(log_test[case][1]))):  # add for each event the difference between this
            if (len(log_test[case][1]) - 1) > i:
                this = event
                following = log_test[case][1][i + 1]
                real_time_tup = (this, following)
                real_diff.append(timeDiff(real_time_tup, datetimeFormat))
            else:
                real_diff.append(0)
        log_test[case].extend([real_diff])

    case_names = []
    event_names = []
    timestamp = []
    p_event = []
    p_timestamp = []
    time_diff = []

    for i in log_test.keys():
        for x in range(len(log_test[i][0])):
            case_names.append(i)

            event_names.append(log_test[i][0][x])
            timestamp.append(log_test[i][1][x])
            p_event.append(log_test[i][2][x])
            p_timestamp.append(log_test[i][3][x])
            time_diff.append(log_test[i][4][x])

    frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
                  'TimeStamp': timestamp, 'Real_TimeDiff': time_diff, 'Baseline_Event': p_event,
                  'Baseline_TimeDiff': p_timestamp}

    predicted_df = pd.DataFrame.from_dict(frame_dict)

    return predicted_df  # baseline check


def combs_algo(data_train, data_test):
    data_train = data_train.sort_values(by=['case concept:name', 'event time:timestamp'])
    data_train['day_of_week'] = data_train['event time:timestamp'].dt.dayofweek

    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'

    # LOAN DATA
    if 'case LoanGoal' in data_train.columns:
        data_train['case LoanGoal'] = data_train['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

        data_test['case LoanGoal'] = data_test['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

    # ROAD DATA
    elif 'Create Fine' in data_train['event concept:name'].values:
        datetimeFormat = '%Y-%m-%d'

    # print(datetimeFormat)

    pt1 = data_train.columns.get_loc('case concept:name') + 1
    pt2 = data_train.columns.get_loc('event concept:name') + 1
    pt3 = data_train.columns.get_loc('event time:timestamp') + 1
    pt4 = data_train.columns.get_loc('day_of_week') + 1

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
            timestamp = parts[pt3]
            day = parts[pt4]

            if caseid not in log:
                log[caseid] = [[], [], []]

            log[caseid][0].append(task)
            log[caseid][1].append(timestamp)
            log[caseid][2].append(day)

    file.close()

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    event_names = list(data_train['event concept:name'].unique())
    event_names.append('New Event')
    combs = []

    for p in itertools.product(event_names, repeat=2):
        combs.append(p)

    pbar = ProgressBar()
    print('[2/18]')
    for i in pbar(log.keys()):  # check
        ID = []
        stamps = []
        for pairID, stamp in zip(pairwise(log[i][0]), pairwise(log[i][1])):
            ID.append(pairID)
            stamps.append(stamp)

        log[i].append(ID)
        log[i].append(stamps)

    pbar = ProgressBar()
    print('[3/18]')

    for i in pbar(log.keys()):  # check
        count = 0
        for perm in log[i][3]:
            index = log[i][3].index(perm)
            if index == 0:
                log[i][3].insert(0, ('New Event', log[i][3][index][0]))

        for count in range(len(log[i][4])):
            stamp = log[i][4][count]
            if count == 0:
                log[i][4].insert(0, (stamp[0], stamp[0]))

            count += 1

    def timeDiff(tupl, datetimeFormat):
        # deleted - datetimeFormat = '%Y-%m-%d'
        diff = datetime.datetime.strptime(tupl[0], datetimeFormat) \
               - datetime.datetime.strptime(tupl[1], datetimeFormat)

        return abs(diff.days)

    def listCount(lst: list):
        cases = list(data_train['event concept:name'].unique())  # list of all unique event names
        cases.append('New Event')
        best = 0
        for x in cases:
            current = lst.count(x)
            if current >= best:
                best = current
                str_best = x
        return str_best

    pbar = ProgressBar()
    print('[4/18]')

    comb_times = {}

    for comb in pbar(combs):  # check
        day_list = [[], [], [], [], [], [], []]

        for case in log.keys():
            if comb in log[case][3]:
                index = log[case][3].index(comb)
                day = int(log[case][2][index])

                if index < (len(log[case][3]) - 1):
                    nxt_event = log[case][3][index + 1][
                        1]  # we need the second item of the tuple, bc item 1 is repeated
                    day_list[day].append(nxt_event)

                elif index == (len(log[case][3]) - 1):
                    nxt_event = 'New Event'
                    day_list[day].append(nxt_event)

            else:
                pass

        comb_times[comb] = day_list

        for i in range(len(day_list)):
            comb_times[comb][i] = listCount(comb_times[comb][i])

    pbar = ProgressBar()
    print('[5/18]')

    for i in pbar(log.keys()):  # check
        # Add the real time differences
        real_diff = []
        for t in log[i][4]:
            real_diff.append(timeDiff(t, datetimeFormat))
        log[i].extend([real_diff])

    """Adding predictions based on the combination with respect to the week. """

    for i in log.keys():  # check
        current = log[i][3]
        prediction = []

        for perm in current:
            index = current.index(perm)
            day = int(log[i][2][index])
            current_prediction = comb_times[perm][day]

            if current_prediction != 0:
                prediction.append(current_prediction)
            else:
                merged_list = list(itertools.chain.from_iterable(comb_times[perm]))
                pred = listCount(merged_list)
                prediction.append(pred)

        log[i].extend([prediction])

        current_real = []

        for x in log[i][0]:
            if log[i][0].index(x) == 0:
                current_real.append('New Event')
            else:
                current_real.append(x)
        log[i].extend([current_real])

    """Storing all time differences for every combination."""

    pbar = ProgressBar()
    print('[6/18]')

    times = {}
    for comb in pbar(combs):  # check
        for case in log.keys():
            if comb in log[case][3]:
                count = log[case][3].index(comb)
                diff = timeDiff(log[case][4][count], datetimeFormat)
                if comb not in times:
                    times[comb] = []
                    times[comb].append(diff)
                else:
                    times[comb].append(diff)
            else:
                pass
        if comb in times.keys():
            times[comb] = int(np.ceil(np.mean(times[comb])))

    ### TEST

    data_test = data_test.sort_values(by=['case concept:name', 'event time:timestamp'])

    data_test['day_of_week'] = data_test['event time:timestamp'].dt.dayofweek

    # ADDITION
    if 'case LoanGoal' in data_test:
        data_test['case LoanGoal'] = data_test['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

    data_test.to_csv("./build/fixed_test.csv")

    t_log = dict()

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

            if caseid not in t_log:
                t_log[caseid] = [[], [], []]

            t_log[caseid][0].append(task)
            t_log[caseid][1].append(timestamp)
            t_log[caseid][2].append(day)
    file.close()

    os.remove('./build/fixed.csv')
    os.remove('./build/fixed_test.csv')

    bugs = []

    for i in t_log.keys():
        if len(t_log[i][0]) == 1:
            bugs.append(i)

    for x in bugs:
        del t_log[x]
        data_test.drop(data_test.index[data_test['case concept:name'] == x], inplace=True)

    m = 0

    for i in log.keys():
        if len(log[i][0]) > m:
            m = len(log[i][0])

    delete = []
    for i in t_log.keys():
        if len(t_log[i][0]) > m:
            delete.append(i)

    for i in delete:
        data_test.drop(data_test.index[data_test['case concept:name'] == i], inplace=True)
        del t_log[i]

    pbar = ProgressBar()
    print('[7/18]')

    for i in pbar(t_log.keys()):  # check
        ID = []
        stamps = []
        for pairID, stamp in zip(pairwise(t_log[i][0]), pairwise(t_log[i][1])):
            ID.append(pairID)
            stamps.append(stamp)

        t_log[i].append(ID)
        t_log[i].append(stamps)

    pbar = ProgressBar()
    print('[8/18]')

    for i in pbar(t_log.keys()):  # check
        count = 0
        for perm in t_log[i][3]:
            index = t_log[i][3].index(perm)
            if index == 0:
                t_log[i][3].insert(0, ('New Event', t_log[i][3][index][0]))

        for count in range(len(t_log[i][4])):
            stamp = t_log[i][4][count]
            if count == 0:
                t_log[i][4].insert(0, (stamp[0], stamp[0]))

            count += 1

    pbar = ProgressBar()
    print('[9/18]')

    for i in pbar(t_log.keys()):  # check
        # Add the real time differences
        real_diff = []
        for t in t_log[i][4]:
            real_diff.append(timeDiff(t, datetimeFormat))
        t_log[i].extend([real_diff])

    """Adding predictions based on the combination with respect to the week. """
    pbar = ProgressBar()
    print('[10/18]')

    for i in pbar(t_log.keys()):  # check
        current = t_log[i][3]
        prediction = []

        for perm in current:
            index = current.index(perm)
            day = int(t_log[i][2][index])
            current_prediction = comb_times[perm][day]

            if current_prediction != 0:
                prediction.append(current_prediction)
            else:
                merged_list = list(itertools.chain.from_iterable(comb_times[perm]))
                pred = listCount(merged_list)
                prediction.append(pred)

        t_log[i].extend([prediction])

        current_real = []

        for x in t_log[i][0]:
            if t_log[i][0].index(x) == 0:
                current_real.append('New Event')
            else:
                current_real.append(x)
        t_log[i].extend([current_real])

    pbar = ProgressBar()
    print('[11/18]')

    """Prediction for time difference, we check whether event is last and then predict 0 for it!"""
    for i in pbar(t_log.keys()):  # check
        time_pred = []
        avg_time_lst = []
        for ev, pred in zip(t_log[i][0], t_log[i][6]):
            last = len(t_log[i][0]) - 1

            if t_log[i][0].index(ev) == last:
                avg_time = sum(avg_time_lst) / len(avg_time_lst)
                time_pred.append(0)

            elif (ev, pred) in times:
                time_perm = times[(ev, pred)]
                avg_time_lst.append(time_perm)

                time_pred.append(times[(ev, pred)])

            elif (ev, pred) not in times:  # we need a better prediction here

                time_pred.append(0)

        avg_time = sum(avg_time_lst) / len(avg_time_lst)

        x = 0
        while x < len(time_pred) - 1:
            time_pred[x] = (time_pred[x] + avg_time) / 2
            x += 1

        t_log[i].extend([time_pred])

    case_names = []
    event_names = []
    timestamp = []
    p_event = []
    current_real = []

    real_diff = []
    pred_diff = []

    for i in t_log.keys():
        for x in range(len(t_log[i][0])):
            case_names.append(i)
            event_names.append(t_log[i][0][x])
            timestamp.append(t_log[i][1][x])
            p_event.append(t_log[i][6][x])
            current_real.append(t_log[i][7][x])

            real_diff.append(t_log[i][5][x])
            pred_diff.append(t_log[i][8][x])

    real_diff.append(0)

    frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
                  'TimeStamp': timestamp, 'Current_Event': current_real,
                  'Real_TimeDiff': real_diff[1:], 'Combs_Event': p_event, 'Combs_TimeDiff': pred_diff}

    predicted_df = pd.DataFrame.from_dict(frame_dict)

    return predicted_df  # comb check


def de_tree(data_train, data_test):
    data_train['event time:timestamp'] = pd.to_datetime(data_train['event time:timestamp'])
    data_train = data_train.sort_values(by=['case concept:name', 'event time:timestamp'])

    data_test['event time:timestamp'] = pd.to_datetime(data_test['event time:timestamp'])
    data_test = data_test.sort_values(by=['case concept:name', 'event time:timestamp'])

    # LOAN DATA
    if 'case LoanGoal' in data_train.columns:
        data_train['case LoanGoal'] = data_train['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

        data_test['case LoanGoal'] = data_test['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)

    pt1 = data_train.columns.get_loc('case concept:name') + 1
    pt2 = data_train.columns.get_loc('event concept:name') + 1
    pt3 = data_train.columns.get_loc('event time:timestamp') + 1

    # DANGER
    cases = list(data_train['event concept:name'].unique()) + list(data_test['event concept:name'].unique())
    cases.append('New Case')
    cases = list(set(cases))

    # 1. Train Data

    # DANGER
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

    for i in log.keys():  # updating the dictionary to contain also all next events
        current = log[i][0]  # recording the cuurent case' events

        real_next = current[1:]  # next real events
        real_next.append('New Case')  # adding a 'new case' as real next event for every last event

        log[i].append(real_next)  # adding the real next events to the log file

    # 2. Test Data

    # DANGER
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

    """Fixing a bug of cases that are in the test data but are incomplete due to the train-test split."""

    bugs = []

    for i in log_test.keys():  # recording the cases which have events cut because of the train - test split
        if len(log_test[i][0]) == 1:
            bugs.append(i)

    for x in bugs:  # deleting the above mentioned events
        del log_test[x]
        data_test.drop(data_test.index[data_test['case concept:name'] == x], inplace=True)

    for i in log_test.keys():
        current = log_test[i][0]  # current case' events

        real_next = current[1:]  # next real events
        real_next.append('New Case')  # adding a 'new case' as real next event for every last event
        log_test[i].append(real_next)

        # 3. Storing the data

    #  new dictionary that will contain for every position(key) the observed traces and next events for each trace(values)
    #  so case [A, B, C] would be saved as {0:[[A],[B]], 1: [[A,B], [C]], 2: [[A, B, C], [New Case]]}
    train_data = {}
    pbar = ProgressBar()
    print('[12/18]')

    for i in pbar(log.keys()):
        count = 0
        for x in log[i][0]:
            case = log[i][0]
            time = log[i][2]  # DANGER

            if count not in train_data:  # making the two lists in the dictionary
                train_data[count] = [[], [],
                                     []]  # list 1 is all for all traces of the position, list 2 is for all next events
            # DANGER EXTRA LIST
            train_data[count][0].append(case[:count + 1])  # appending the trace
            train_data[count][2].append(time[count])  # DANGER
            if count < len(case) - 1:
                train_data[count][1].append(case[count + 1])  # appending the next event of the trace

            elif count == len(case) - 1:
                train_data[count][1].append('New Case')

            count += 1

    #  repeating the same process on the test data
    test_data = {}
    pbar = ProgressBar()
    print('[13/18]')
    for i in pbar(log_test.keys()):
        count = 0
        for x in log_test[i][0]:
            case = log_test[i][0]
            time = log_test[i][2]  # DANGER

            if count not in test_data:
                test_data[count] = [[], [], []]

            test_data[count][0].append(case[:count + 1])  # appending the trace
            test_data[count][2].append(time[count])  # DANGER

            if count < len(case) - 1:
                test_data[count][1].append(case[count + 1])  # appending the next event of the trace

            elif count == len(case) - 1:
                test_data[count][1].append('New Case')

            count += 1

    # 4. Encoding

    # encoding all unique event names of all the data into integers

    le = preprocessing.LabelEncoder()
    le.fit(cases)  # encoding all event names into integers

    ### 4.1 TRAIN

    pbar = ProgressBar()
    print('[14/18]')

    for i in pbar(train_data.keys()):  # the dictionaries from above are encoded into integers

        encoded = []
        for trace in train_data[i][0]:  # encoding all strings of a trace, can be multiple if case lenght is more than 2
            local_encoded = []
            for event in trace:
                local_encoded.append(int(le.transform([event])))  # transforming into integer
            encoded.append(local_encoded)

        train_data[i][0] = np.array(encoded)  # making the list with integers into array so the tree can take it

        encoded_next = []  # encoding all strings of next events for a trace, its always length 1 !
        for g in train_data[i][1]:
            encoded_next.append(int(le.transform([g])))  # transforming into integer

        train_data[i][1] = np.array(encoded_next)  # making the list with integers into array

    ### 4.2 TEST

    # repeating the procedure from above on the test data

    pbar = ProgressBar()
    print('[15/18]')

    for i in pbar(test_data.keys()):

        encoded = []
        for trace in test_data[i][0]:
            local_encoded = []
            for event in trace:
                local_encoded.append(int(le.transform([event])))
            encoded.append(local_encoded)

        test_data[i][0] = np.array(encoded)

        encoded_next = []
        for g in test_data[i][1]:
            encoded_next.append(int(le.transform([g])))

        test_data[i][1] = np.array(encoded_next)

    # 5. Training the decision tree

    # Function for training decision tree for any given position (as long as the position is in the train data)

    # DANGER

    def decision_tree(pos):
        x_train = train_data[pos][0]
        # combine full trace data with weekday
        x_week = np.array(train_data[pos][2]).reshape(-1, 1)
        x_new = np.concatenate((x_train, x_week), axis=1)
        y_train = train_data[pos][1]

        classifier = DecisionTreeClassifier()
        classifier.fit(x_new, y_train)

        return classifier

    predictors = {}  # dictionary to contain all decision trees given the position
    #  key - position, value - decision tree for that position
    pbar = ProgressBar()
    print('[16/18]')

    for i in pbar(test_data.keys()):
        if i > len(train_data) - 1:
            predictors[i] = decision_tree(len(train_data) - 1)

        else:
            predictors[i] = decision_tree(i)

    # 6. Adding predictions

    pbar = ProgressBar()
    print('[17/18]')

    for i in pbar(
            log_test.keys()):  # adding an array with the encoding to the log_test dict. for every case in the test
        current = log_test[i][0]

        encoded = []  # list will contain all event names encoded into integers
        for g in current:
            encoded.append(int(le.transform([g])))
        encoded = np.array(encoded)
        log_test[i].append(encoded)

    def update_tree(case):

        case = case.tolist()

        count = 0
        for x in case:  # case is an array
            if count not in train_data:  # making the two lists in the dictionary
                # list 1 is all for all traces of the position, list 2 is for all next events
                train_data[count] = [np.array([]), np.array([])]
            train_data[count][0] = train_data[count][0].tolist()
            train_data[count][1] = train_data[count][1].tolist()

            train_data[count][0].append(case[:count + 1])  # appending the trace

            if count < len(case) - 1:
                train_data[count][1].append(case[count + 1])  # appending the next event of the trace

            elif count == len(case) - 1:
                train_data[count][1].append(int(le.transform(['New Case'])))

            train_data[count][0] = np.array(train_data[count][0])
            train_data[count][1] = np.array(train_data[count][1])

            predictors[count] = decision_tree(count)

            count += 1

    pbar = ProgressBar()
    print('[18/18]')

    for i in pbar(log_test.keys()):  # making predictions for every case in the log_test dict

        current_encoded = log_test[i][4]  # DANGER
        times = log_test[i][1]
        weeks = log_test[i][2]
        predictions = []  # list that will contain all predictions for a given case
        count = 0

        for x in current_encoded:

            # the if-else is a checks whether the case length is more than any case length observed in the train data
            if count >= len(train_data) - 1:  # if its in the train data we call the appropriate decision tree

                tree = predictors[len(train_data) - 1]
                p_trace = current_encoded[:(len(train_data))].reshape(-1, len(train_data))
                # Create new array with full trace and weekdays, same as with train
                p_weeks = np.array(weeks[len(train_data) - 1]).reshape(-1, 1)
                p_new = np.concatenate((p_trace, p_weeks), axis=1)
                pred = tree.predict(p_new)
                pred_string = le.inverse_transform(pred)[0]
                predictions.append(pred_string)



            else:  # if its not in the train data then we use the last observed decision tree from the train data

                tree = predictors[count]
                p_trace = current_encoded[:count + 1].reshape(-1, count + 1)
                # same as above
                p_weeks = np.array(weeks[count]).reshape(-1, 1)
                p_new = np.concatenate((p_trace, p_weeks), axis=1)
                pred = tree.predict(p_new)
                pred_string = le.inverse_transform(pred)[0]
                predictions.append(pred_string)

            count += 1

        log_test[i].append(predictions)  # adding all predictions to the log_test of the current case

        # UNCOMMENT THE LINE BELOW FOR ONLINE TRAINING

        # update_tree(current_encoded)

    # 7. Evaluation

    # making lists for every column we will have in the frame
    case_names = []
    event_names = []
    timestamp = []
    p_event = []
    current_real = []

    for i in log_test.keys():  # appending the right things to every list from the log_test file
        for x in range(len(log_test[i][0])):
            case_names.append(i)
            event_names.append(log_test[i][0][x])
            timestamp.append(log_test[i][1][x])
            p_event.append(log_test[i][-1][x])
            current_real.append(log_test[i][3][x])

    frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
                  'TimeStamp': timestamp, 'Next_Event': current_real, 'DTree_Event': p_event}
    predicted_df = pd.DataFrame.from_dict(frame_dict)

    ### TIMESTAMPS REGRESSION

    print("Timestamp Linear Regression Prediction...")

    train = data_train.copy()
    test = data_test.copy()

    # Add new useful columns for the model train
    train['position_event'] = train.groupby('case concept:name').cumcount()
    train['position_event'] = train['position_event'] + 1
    train['week_day'] = train['event time:timestamp'].dt.dayofweek

    # Encoding all event names into integers
    cases = train['event concept:name'].unique().tolist()
    cases.insert(0, 'New Case')
    le_case = preprocessing.LabelEncoder()
    le_case.fit(cases)

    # Encoding lifecycle into integers
    life = train['event lifecycle:transition'].unique().tolist()
    le_life = preprocessing.LabelEncoder()
    le_life.fit(life)

    # Preprocess data for model train
    # Event poistion
    x_train_position = np.array(train['position_event']).reshape(-1, 1)[:]
    # Previous event
    x_train_prev = list(train['event concept:name'])
    x_train_prev = le_case.transform(x_train_prev)
    x_train_prev = np.array(x_train_prev).reshape(-1, 1)[:]
    # Event
    x_train_event = list(train['event concept:name'])
    x_train_event.insert(len(train), 'New Case')
    x_train_event = le_case.transform(x_train_event)
    x_train_event = np.array(x_train_event).reshape(-1, 1)[1:]
    # Day of the week previous event event
    x_train_week = list(train['week_day'])
    x_train_week = np.array(x_train_week).reshape(-1, 1)[:]
    # Timestamp event
    train[['event time:timestamp']] = train[['event time:timestamp']].astype(str)
    x_train_date = list(train['event time:timestamp'])
    x_train_date.insert(len(train), None)
    x_train_date = np.array(x_train_date).reshape(-1, 1)[1:]
    # Timestamp previous event
    x_train_date_prev = list(train['event time:timestamp'])
    x_train_date_prev = np.array(x_train_date_prev).reshape(-1, 1)[:]
    # Event Lifecycle
    x_train_life = list(train['event lifecycle:transition'])
    x_train_life = le_life.transform(x_train_life)
    x_train_life = np.array(x_train_life).reshape(-1, 1)[:]


    # Length case for train set
    cases = train.groupby(['case concept:name'])
    per_case = pd.DataFrame({'no of events': cases['eventID '].count()})
    lst_per_case = per_case["no of events"].tolist()
    case_length = []
    for length in lst_per_case:
        case_length.extend(repeat(length, length))
    x_train_length_case = np.array(case_length).reshape(-1, 1)[:]


    # Combine features for the model train
    x_train_new = np.concatenate((x_train_position, x_train_prev, x_train_event, x_train_week, x_train_date,
                                  x_train_date_prev, x_train_length_case, x_train_life), axis=1)

    # Add features to new dataframe train
    df_train = pd.DataFrame(data=x_train_new,
                            columns=['position_event', 'prev_event', 'event', 'week_day_prev', 'date', 'date_prev',
                                     'case_length', 'lifecycle'])
    df_train.loc[df_train['position_event'] == df_train['case_length'], 'event'] = int(le.transform(['New Case']))
    df_train[['date', 'date_prev']] = df_train[['date', 'date_prev']].apply(pd.to_datetime)
    df_train.loc[df_train['event'] == int(le.transform(['New Case'])), 'date'] = None
    df_train['in_between'] = (df_train['date'] - df_train['date_prev']).dt.total_seconds()
    df_train.loc[df_train['event'] == int(le.transform(['New Case'])), 'in_between'] = 0
    df_train[['case_length']] = df_train[['case_length']].astype(int)

    # Train Dummies

    # Implementing dummies train
    df_train = pd.get_dummies(df_train, columns=['event', 'prev_event', 'week_day_prev', 'position_event', 'lifecycle'])
    df_train = df_train.drop(['date', 'date_prev'], 1)

    # Test Data Preprocessing

    # Add new useful columns for the model test
    test['position_event'] = test.groupby('case concept:name').cumcount()
    test['position_event'] = test['position_event'] + 1
    test['week_day'] = test['event time:timestamp'].dt.dayofweek
    predicted_events = predicted_df['DTree_Event'][:].tolist()
    test['pred_event'] = predicted_events

    # Preprocess data for model test
    # Event poistion
    x_test_position = np.array(test['position_event']).reshape(-1, 1)[:]
    # Previous event
    x_test_prev = test['event concept:name'].tolist()
    x_test_prev = le_case.transform(x_test_prev)
    x_test_prev = np.array(x_test_prev).reshape(-1, 1)[:]
    # Predicted Event
    x_test_event = test['pred_event'].tolist()
    x_test_event = le_case.transform(x_test_event)
    x_test_event = np.array(x_test_event).reshape(-1, 1)[:]
    # Day of the week previous event
    x_test_week = test['week_day'].tolist()
    x_test_week = np.array(x_test_week).reshape(-1, 1)[:]
    # Timestamp event
    test[['event time:timestamp']] = test[['event time:timestamp']].astype(str)
    x_test_date = list(test['event time:timestamp'])
    x_test_date.insert(len(test), None)
    x_test_date = np.array(x_test_date).reshape(-1, 1)[1:]
    # Timestamp previous event
    x_test_date_prev = list(test['event time:timestamp'])
    x_test_date_prev = np.array(x_test_date_prev).reshape(-1, 1)[:]
    # Event Lifecycle
    x_test_life = test['event lifecycle:transition'].tolist()
    x_test_life = le_life.transform(x_test_life)
    x_test_life = np.array(x_test_life).reshape(-1, 1)[:]


    # Length case for test set
    test_cases = test.groupby(['case concept:name'])
    per_case_test = pd.DataFrame({'no of events': test_cases['eventID '].count()})
    lst_per_case_test = per_case_test["no of events"].tolist()
    case_length_test = []
    for length in lst_per_case_test:
        case_length_test.extend(repeat(length, length))
    x_test_length_case = np.array(case_length_test).reshape(-1, 1)[:]

    # Combine features for the model test
    x_test_new = np.concatenate((x_test_position, x_test_prev, x_test_event, x_test_week, x_test_date, x_test_date_prev,
                                 x_test_length_case, x_test_life), axis=1)

    # Add features to new dataframe test
    df_test = pd.DataFrame(data=x_test_new,
                           columns=['position_event', 'prev_event', 'event', 'week_day_prev', 'date', 'date_prev',
                                    'case_length', 'lifecycle'])
    df_test.loc[df_test['position_event'] == df_test['case_length'], 'date'] = None
    df_test[['date', 'date_prev']] = df_test[['date', 'date_prev']].apply(pd.to_datetime)
    df_test['in_between'] = (df_test['date'] - df_test['date_prev']).dt.total_seconds()
    df_test.loc[df_test['position_event'] == df_test['case_length'], 'in_between'] = 0
    df_test[['case_length']] = df_test[['case_length']].astype(int)

    # Remove cases with more events than the cases in the train set
    df_test = df_test[df_test['case_length'] <= max(df_train['case_length'])]

    # Test Dummies

    # Implementing dummies test
    df_test_d = pd.get_dummies(df_test, columns=['event', 'prev_event', 'week_day_prev', 'position_event', 'lifecycle'])
    df_test_d = df_test_d.drop(['date', 'date_prev'], 1)

    # Feature selection and model training

    col_train = df_train.columns
    col_test = df_test_d.columns
    features = set(col_train).intersection(col_test)
    features.discard('in_between')
    X_train = df_train[features]  # Features
    y_train = df_train['in_between']  # Target variable
    X_test = df_test_d[features]  # Features
    y_test = df_test_d['in_between']  # Target variable

    # Training the algorithm
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Evaluation

    # Workaround to get rid of negative values
    # All 'New Case' to 0 and others to absolute value: RMSE 166.7941 (days)
    y_pred = regressor.predict(X_test)
    df_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_pred_fin = pd.concat([df_test, df_predictions], axis=1)
    df_pred_fin.loc[df_pred_fin['event'] == int(le.transform(['New Case'])), 'Predicted'] = 0
    y_pred = df_pred_fin['Predicted']
    y_pred = abs(y_pred)
    df_predictions = pd.DataFrame({'DTree_Actual_TimeDiff': y_test, 'DTree_TimeDiff': y_pred})

    predicted_df = pd.concat([predicted_df, df_predictions], axis=1)

    predicted_df = predicted_df[predicted_df['DTree_Actual_TimeDiff'].notna()]
    predicted_df = predicted_df[predicted_df['DTree_TimeDiff'].notna()]

    return predicted_df


if __name__ == '__main__':
    train = pd.read_csv('./data/road-train-pre.csv')
    test = pd.read_csv('./data/road-test-pre.csv')

    base = baseline(train, test)
    print('Baseline prediction Done')

    base_perm = combs_algo(train, test)
    print('Permutation prediction Done')

    # Adding baseline predictions
    ones = list(set(base['Case_ID']) - set(base_perm['Case_ID']))

    base_pred_ev = base['Baseline_Event']
    base_pred_ts = base['Baseline_TimeDiff']

    for case in ones:
        row = base[base['Case_ID'] == case].index
        base = base.drop(row)

    base_perm['Baseline_Event'] = base_pred_ev
    base_perm['Baseline_TimeDiff'] = base_pred_ts

    base_perm.to_csv('./build/base_perm_test.csv', index=False)
