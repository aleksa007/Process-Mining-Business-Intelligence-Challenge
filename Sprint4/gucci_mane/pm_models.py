import math
import pandas as pd
import numpy as np
import itertools
from itertools import tee, repeat
from progressbar import ProgressBar
import os, datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, metrics

def baseline(data_train, data_test):
    (eventID, caseID, event_name, transition, stamp) = ("eventID ", "case concept:name",
                                                        "event concept:name", "event lifecycle:transition",
                                                        "event time:timestamp")

    # Sorting the frame chronologically and per case.
    data_train = data_train.sort_values(by=[caseID, stamp])
    data_test = data_test.sort_values(by=[caseID, stamp])

    # New .csv file with the fixed sorting and timestamp
    data_train.to_csv("./build/fixed_train.csv")
    data_test.to_csv("./build/fixed_test.csv")
    # Creating a dictionary file where the case names (IDs) are the keys and the values are a list of two sublists.
    # The first sublist contains all event names of a case, sorted chronologically.
    # The second sublist contains all event time stamps of a case, sorted chronologically.

    # file = open('fixed.csv', 'r')
    log = dict()
    # print('making dictionary from fixed file')
    with open('./build/fixed_train.csv', 'r') as f:
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
    # print('making test dictionary from fixed file')
    with open('./build/fixed_test.csv', 'r') as f:
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

    # print('done making dictionaries')
    # print('deleting fixed files')
    os.remove('./build/fixed_train.csv')
    os.remove('./build/fixed_test.csv')

    # print('done')

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
    #print('Event prediction done')
    ### Timestamp prediction

    # list of lists that contains all timestamps for all events of a case, where every sublist is a new case
    time_list = []
    for case in log.keys():
        time_list.append((log[case][1]))

    times_longest_event = max(map(len, time_list))  # longest event
    # print("Longest event in train: ", times_longest_event)

    time_list = []
    for case in log_test.keys():
        time_list.append((log_test[case][1]))

    times_longest_event_test = max(map(len, time_list))  # longest event

    # print("Longest event in test: ", times_longest_event_test)

    def timeDiff(tup):
        """The function calculates the difference between two timestamps and returns the absolute value."""

        datetimeFormat = '%Y-%m-%d'
        diff = datetime.datetime.strptime(tup[0], datetimeFormat) \
               - datetime.datetime.strptime(tup[1], datetimeFormat)

        return abs(diff.days)

    pbar = ProgressBar()

    # list of lists that contains all event timestamps for every position, where each sublist is a new position
    pos_times = []
    print('[1/13]')
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
        if not math.isnan(avg_pos):  # a small check to avoid bugs because the last value is a NaN
            avg_pos = int(np.mean(i))
            time_frequency.append(avg_pos)
        else:
            time_frequency.append(avg_pos)
    #print('Time prediction Done')

    ### Adding predictions to the file
    # * To the test file
    for case in log_test.keys():
        n_events = len(log_test[case][0])
        n_times = len(log_test[case][1])

        if len(log_test[case][0]) > times_longest_event:
            n_unknown = len(log_test[case][0]) - times_longest_event
            evs = event_frequency[:n_events]
            evs.extend(['Unknown' for one in range(0, n_unknown)])

            tms = time_frequency[:n_times]
            tms.extend([31 for one in range(0, n_unknown)])

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
                real_diff.append(timeDiff(real_time_tup))
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
                  'TimeStamp': timestamp, 'Real_TimeDiff': time_diff, 'Baseline_Event': p_event,
                  'Baseline_TimeDiff': p_timestamp}

    predicted_df = pd.DataFrame.from_dict(frame_dict)

    return predicted_df

def combs_algo(data_train, data_test):

    data_train = data_train.sort_values(by=['case concept:name', 'event time:timestamp'])
    data_train['day_of_week'] = data_train['event time:timestamp'].dt.dayofweek

    ## ADDITION
    if 'case LoanGoal' in data_train.columns:
        data_train['case LoanGoal'] = data_train['case LoanGoal'].apply(
            lambda x: 'Other - see explanation' if x == 'Other, see explanation' else x)
        datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'

        pt1, pt2, pt3, pt4 = 4, 8, 12, 13
    else:  # check if raod data
        datetimeFormat = '%Y-%m-%d'

        pt1, pt2, pt3, pt4 = 2, 3, 5, 6

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
    print('[2/13]')
    for i in pbar(log.keys()):
        ID = []
        stamps = []
        for pairID, stamp in zip(pairwise(log[i][0]), pairwise(log[i][1])):
            ID.append(pairID)
            stamps.append(stamp)

        log[i].append(ID)
        log[i].append(stamps)

    pbar = ProgressBar()
    print('[3/13]')

    for i in pbar(log.keys()):
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
    print('[4/13]')

    comb_times = {}

    for comb in pbar(combs):
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
    print('[5/13]')

    for i in pbar(log.keys()):
        # Add the real time differences
        real_diff = []
        for t in log[i][4]:
            real_diff.append(timeDiff(t, datetimeFormat))
        log[i].extend([real_diff])

    """Adding predictions based on the combination with respect to the week. """

    for i in log.keys():
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
    print('[6/13]')

    times = {}
    for comb in pbar(combs):
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

    # Convert from string to datetime
    #data_test['event time:timestamp'] = pd.to_datetime(data_test['event time:timestamp'])

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
            #print(i)
            #print(t_log[i])
            delete.append(i)
            # m_t = len(log_test[i][0])

    for i in delete:

        data_test.drop(data_test.index[data_test['case concept:name'] == i], inplace=True)

        del t_log[i]


    pbar = ProgressBar()
    print('[7/13]')

    for i in pbar(t_log.keys()):
        ID = []
        stamps = []
        for pairID, stamp in zip(pairwise(t_log[i][0]), pairwise(t_log[i][1])):
            ID.append(pairID)
            stamps.append(stamp)

        t_log[i].append(ID)
        t_log[i].append(stamps)

    pbar = ProgressBar()
    print('[8/13]')

    for i in pbar(t_log.keys()):
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
    print('[9/13]')

    for i in pbar(t_log.keys()):
        # Add the real time differences
        real_diff = []
        for t in t_log[i][4]:
            real_diff.append(timeDiff(t, datetimeFormat))
        t_log[i].extend([real_diff])

    """Adding predictions based on the combination with respect to the week. """

    for i in t_log.keys():
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

    """Prediction for time difference, we check whether event is last and then predict 0 for it!"""
    for i in t_log.keys():
        time_pred = []
        for ev, pred, day in zip(t_log[i][0], t_log[i][6], t_log[i][2]):
            last = len(t_log[i][0]) - 1

            if t_log[i][0].index(ev) == last:
                time_pred.append(0)
            elif pred == 'New Event':
                time_pred.append(0)
            elif (ev, pred) in times:
                if int(day) + times[(ev, pred)] % 7 == 5:
                    time_pred.append(times[(ev, pred)] + 2)
                elif int(day) + times[(ev, pred)] % 7 == 6:
                    time_pred.append(times[(ev, pred)] + 1)
                else:
                    time_pred.append(times[(ev, pred)])

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

    #print('Len of Comb Algo: {}'.format(len(predicted_df)))


    return predicted_df

def de_tree(data_train, data_test):

    data_train = data_train.sort_values(by=['case concept:name', 'event time:timestamp'])
    data_test = data_test.sort_values(by=['case concept:name', 'event time:timestamp'])

    data_train.to_csv("./build/fixed.csv")

    log = dict()  # dictionary that contains all information for a case - key: case name; values: events, timestamps
    with open('./build/fixed.csv', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[2]

            task = parts[3]
            timestamp = parts[5]

            if caseid not in log:
                log[caseid] = [[], []]

            log[caseid][0].append(task)  # adding the events as a list into the dictionary
            log[caseid][1].append(timestamp)  # adding the timestamps as a list into the dictionary

    file.close()

    os.remove('./build/fixed.csv')

    for i in log.keys():  # updating the dictionary to contain also all next events
        current = log[i][0]  # recording the cuurent case' events

        real_next = current[1:]  # next real events
        real_next.append('New Case')  # adding a 'new case' as real next event for every last event

        log[i].append(real_next)  # adding the real next events to the log file

    #  Repeating the same process from above on the test data.

    data_test.to_csv("./build/fixed_test.csv")

    log_test = dict()
    with open('./build/fixed_test.csv', 'r') as file:
        next(file)
        for line in file:
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

    m = 0

    for i in log.keys():
        if len(log[i][0]) > m:
            m = len(log[i][0])


    delete = []
    for i in log_test.keys():
        if len(log_test[i][0]) > m:
            delete.append(i)

    for i in delete:
        data_test.drop(data_test.index[data_test['case concept:name'] == i], inplace=True)

        del log_test[i]

    #  new dictionary that will contain for every position(key) the observed traces and next events for each trace(values)
    #  so case [A, B, C] would be saved as {0:[[A],[B]], 1: [[A,B], [C]], 2: [[A, B, C], [New Case]]}
    train_data = {}

    for i in log.keys():
        count = 0
        for x in log[i][0]:
            case = log[i][0]
            # ind = log[i][0].index(x)

            if count not in train_data:  # making the two lists in the dictionary
                train_data[count] = [[],
                                     []]  # list 1 is all for all traces of the position, list 2 is for all next events

            train_data[count][0].append(case[:count + 1])  # appending the trace

            if count < len(case) - 1:
                train_data[count][1].append(case[count + 1])  # appending the next event of the trace

            elif count == len(case) - 1:
                train_data[count][1].append('New Case')

            count += 1

    #  repeating the same process on the test data
    test_data = {}

    for i in log_test.keys():
        count = 0
        for x in log_test[i][0]:
            case = log_test[i][0]
            # ind = log_test[i][0].index(x)

            if count not in test_data:
                test_data[count] = [[], []]

            test_data[count][0].append(case[:count + 1])  # appending the trace

            if count < len(case) - 1:
                test_data[count][1].append(case[count + 1])  # appending the next event of the trace

            elif count == len(case) - 1:
                test_data[count][1].append('New Case')

            count += 1

    # encoding all unique event names of all the data into integers

    cases = list(data_train['event concept:name'].unique()) + list(
        data_test['event concept:name'].unique())  # all events
    cases.append('New Case')  # adding the 'New Case' because we predict next event is going to be new case
    cases = list(set(cases))
    le = preprocessing.LabelEncoder()
    le.fit(cases)  # encoding all event names into integers

    pbar = ProgressBar()
    print('[10/13]')

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

    # repeating the procedure from above on the test data

    pbar = ProgressBar()
    print('[11/13]')

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

    # Function for training decision tree for any given position (as long as the position is in the train data)

    def decision_tree(pos):

        x_train = train_data[pos][0]
        y_train = train_data[pos][1]

        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)

        return classifier

    predictors = {}  # dictionary to contain all decision trees given the position
    #  key - position, value - decision tree for that position

    for i in test_data.keys():
        if i > len(train_data) - 1:
            predictors[i] = decision_tree(len(train_data) - 1)

        else:
            predictors[i] = decision_tree(i)

    pbar = ProgressBar()
    print('[12/13]')

    for i in pbar(
            log_test.keys()):  # adding an array with the encoding to the log_test dict. for every case in the test
        current = log_test[i][0]

        encoded = []  # list will contain all event names encoded into integers
        for g in current:
            encoded.append(int(le.transform([g])))
        encoded = np.array(encoded)
        log_test[i].append(encoded)

    pbar = ProgressBar()
    print('[13/13]')

    for i in pbar(log_test.keys()):  # making predictions for every case in the log_test dict

        current_encoded = log_test[i][3]
        predictions = []  # list that will contain all predictions for a given case
        count = 0

        for x in current_encoded:

            # the if-else is a checks whether the case length is more than any case length observed in the train data
            if count >= len(train_data) - 1:  # if its in the train data we call the appropriate decision tree

                tree = predictors[len(train_data) - 1]  # calling the right tree given the position
                p = current_encoded[:(len(train_data))]  # taking the trace
                p = p.reshape(1, -1)
                pred = tree.predict(p)  # making a prediction
                pred_string = le.inverse_transform(pred)[0]  # transforming the prediction into a string
                predictions.append(pred_string)  # appending the prediction as a string to the log_test data



            else:  # if its not in the train data then we use the last observed decision tree from the train data

                tree = predictors[count]  # calling the right tree given the position
                p = current_encoded[:count + 1]  # taking the trace
                p = p.reshape(1, -1)  # we need to do that, idk why
                pred = tree.predict(p)  # making a prediction
                pred_string = le.inverse_transform(pred)[0]  # transforming the prediction into a string
                predictions.append(pred_string)  # appending the prediction as a string to the log_test data

            count += 1

        log_test[i].append(predictions)  # adding all predictions to the log_test of the current case

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
            p_event.append(log_test[i][4][x])
            current_real.append(log_test[i][2][x])

    # dictionary that will be used to make the frame
    frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
                  'TimeStamp': timestamp, 'Next_Event': current_real, 'DTree_Event': p_event}

    predicted_df = pd.DataFrame.from_dict(frame_dict)  # making a frame

    #print('Len of DTree Algo Event Pred: {}'.format(len(predicted_df)))

    # TODO: TIMESTAMPS REGRESSION

    data_train['position_event'] = data_train.groupby('case concept:name').cumcount()
    data_train['position_event'] = data_train['position_event'] + 1
    data_train['week_day'] = data_train['event time:timestamp'].dt.dayofweek

    # Encoding all event names into integers
    cases = data_train['event concept:name'].unique().tolist()
    cases.insert(0, 'New Case')
    le_case = preprocessing.LabelEncoder()
    le_case.fit(cases)

    # Encoding lifecycle into integers
    life = data_train['event lifecycle:transition'].unique().tolist()
    le_life = preprocessing.LabelEncoder()
    le_life.fit(life)

    # Preprocess data for model train
    # Event poistion
    x_train_position = np.array(data_train['position_event']).reshape(-1, 1)[:]
    # Previous event
    x_train_prev = list(data_train['event concept:name'])
    x_train_prev = le_case.transform(x_train_prev)
    x_train_prev = np.array(x_train_prev).reshape(-1, 1)[:]
    # Event
    x_train_event = list(data_train['event concept:name'])
    x_train_event.insert(len(data_train), 'New Case')
    x_train_event = le_case.transform(x_train_event)
    x_train_event = np.array(x_train_event).reshape(-1, 1)[1:]
    # Day of the week previous event event
    x_train_week = list(data_train['week_day'])
    x_train_week = np.array(x_train_week).reshape(-1, 1)[:]
    # Timestamp event
    data_train[['event time:timestamp']] = data_train[['event time:timestamp']].astype(str)
    x_train_date = list(data_train['event time:timestamp'])
    x_train_date.insert(len(data_train), None)
    x_train_date = np.array(x_train_date).reshape(-1, 1)[1:]
    # Timestamp previous event
    x_train_date_prev = list(data_train['event time:timestamp'])
    x_train_date_prev = np.array(x_train_date_prev).reshape(-1, 1)[:]
    # Event Lifecycle
    x_train_life = list(data_train['event lifecycle:transition'])
    x_train_life = le_life.transform(x_train_life)
    x_train_life = np.array(x_train_life).reshape(-1, 1)[:]

    # Length case for train set
    cases = data_train.groupby(['case concept:name'])
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
    df_train.loc[df_train['position_event'] == df_train['case_length'], 'event'] = 5
    df_train[['date', 'date_prev']] = df_train[['date', 'date_prev']].apply(pd.to_datetime)
    df_train.loc[df_train['event'] == 5, 'date'] = None
    df_train['in_between'] = (df_train['date'] - df_train['date_prev']).dt.days
    df_train.loc[df_train['event'] == 5, 'in_between'] = 0
    #df_train

    # Train Dummies

    # Implementing dummies train
    df_train = pd.get_dummies(df_train, columns=['event', 'prev_event', 'week_day_prev', 'position_event', 'lifecycle'])
    df_train = df_train.drop(['date', 'date_prev'], 1)
    #df_train

    # Test Data Preprocessing

    # Add new useful columns for the model test
    data_test['position_event'] = data_test.groupby('case concept:name').cumcount()
    data_test['position_event'] = data_test['position_event'] + 1
    data_test['week_day'] = data_test['event time:timestamp'].dt.dayofweek

    predicted_events = predicted_df['DTree_Event'][:].tolist()
    data_test['pred_event'] = predicted_events

    # Preprocess data for model test
    # Event poistion
    x_test_position = np.array(data_test['position_event']).reshape(-1, 1)[:]
    # Previous event
    x_test_prev = data_test['event concept:name'].tolist()
    x_test_prev = le_case.transform(x_test_prev)
    x_test_prev = np.array(x_test_prev).reshape(-1, 1)[:]
    # Predicted Event
    x_test_event = data_test['pred_event'].tolist()
    x_test_event = le_case.transform(x_test_event)
    x_test_event = np.array(x_test_event).reshape(-1, 1)[:]
    # Day of the week previous event
    x_test_week = data_test['week_day'].tolist()
    x_test_week = np.array(x_test_week).reshape(-1, 1)[:]
    # Timestamp event
    data_test[['event time:timestamp']] = data_test[['event time:timestamp']].astype(str)
    x_test_date = list(data_test['event time:timestamp'])
    x_test_date.insert(len(data_test), None)
    x_test_date = np.array(x_test_date).reshape(-1, 1)[1:]
    # Timestamp previous event
    x_test_date_prev = list(data_test['event time:timestamp'])
    x_test_date_prev = np.array(x_test_date_prev).reshape(-1, 1)[:]
    # Event Lifecycle
    x_test_life = data_test['event lifecycle:transition'].tolist()
    x_test_life = le_life.transform(x_test_life)
    x_test_life = np.array(x_test_life).reshape(-1, 1)[:]

    # Length case for test set
    test_cases = data_test.groupby(['case concept:name'])
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
    df_test['in_between'] = (df_test['date'] - df_test['date_prev']).dt.days
    df_test.loc[df_test['position_event'] == df_test['case_length'], 'in_between'] = 0
    #df_test

    # Remove cases with more events than the cases in the train set
    df_test = df_test[df_test['case_length'] <= max(df_train['case_length'])]
    #df_test

    # Test Dummies

    # Implementing dummies test
    df_test = pd.get_dummies(df_test, columns=['event', 'prev_event', 'week_day_prev', 'position_event', 'lifecycle'])
    df_test = df_test.drop(['date', 'date_prev'], 1)
    #df_test

    # Feature selection and model training


    col_train = df_train.columns
    col_test = df_test.columns
    features = set(col_train).intersection(col_test)
    features.discard('in_between')
    X_train = df_train[features]  # Features
    y_train = df_train['in_between']  # Target variable
    X_test = df_test[features]  # Features
    y_test = df_test['in_between']  # Target variable

    # Training the algorithm
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)


    #print(regressor.intercept_)
    #print(regressor.coef_)

    # Evaluation


    y_pred = regressor.predict(X_test)
    #print('Len of DTree y_pred: {}'.format(len(y_pred)))

    predicted_df['DTree_TimeDiff'] = y_pred

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

    base_perm.to_csv('./build/base_perm_test.csv', index = False)