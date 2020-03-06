import pandas as pd
import numpy as np
import os
from itertools import tee, combinations, permutations
import itertools

from memory_profiler import profile
from progressbar import ProgressBar
import datetime
import time
import argparse
@profile
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

    data_train = pd.read_csv(train_path, error_bad_lines=False)
    data_test = pd.read_csv(test_path, error_bad_lines=False)

    data_train['event time:timestamp'] = pd.to_datetime(data_train['event time:timestamp'])
    data_train = data_train.sort_values(by=['case concept:name', 'event time:timestamp'])
    data_train['day_of_week'] = data_train['event time:timestamp'].dt.dayofweek
    data_train.to_csv("./fixed.csv")

    log = dict()
    with open('./fixed.csv', 'r') as file:
        next(file)
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            caseid = parts[2]

            task = parts[3]
            timestamp = parts[5]
            day = parts[6]

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

    for i in pbar(log.keys()):
        ID = []
        stamps = []
        for pairID, stamp in zip(pairwise(log[i][0]), pairwise(log[i][1])):
            ID.append(pairID)
            stamps.append(stamp)

        log[i].append(ID)
        log[i].append(stamps)

    pbar = ProgressBar()

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

    def timeDiff(tupl):

        datetimeFormat = '%Y-%m-%d'
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

    for i in pbar(log.keys()):
        # Add the real time differences
        real_diff = []
        for t in log[i][4]:
            real_diff.append(timeDiff(t))
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

    times = {}
    for comb in pbar(combs):
        for case in log.keys():
            if comb in log[case][3]:
                count = log[case][3].index(comb)
                diff = timeDiff(log[case][4][count])
                if comb not in times:
                    times[comb] = []
                    times[comb].append(diff)
                else:
                    times[comb].append(diff)
            else:
                pass
        if comb in times.keys():
            times[comb] = int(np.ceil(np.mean(times[comb])))

    # """Prediction for time difference, we check whether event is last and then predict 0 for it!"""
    # for i in log.keys():
    #     time_pred = []
    #     for ev, pred, day in zip(log[i][0], log[i][6], log[i][2]):
    #         last = len(log[i][0]) - 1
    #
    #         if log[i][0].index(ev) == last:
    #             time_pred.append(0)
    #         elif pred == 'New Event':
    #             time_pred.append(0)
    #         elif (ev, pred) in times:
    #             if int(day) + times[(ev, pred)] % 7 == 5:
    #                 time_pred.append(times[(ev, pred)] + 2)
    #             elif int(day) + times[(ev, pred)] % 7 == 6:
    #                 time_pred.append(times[(ev, pred)] + 1)
    #             else:
    #                 time_pred.append(times[(ev, pred)])
    #
    #     log[i].extend([time_pred])
    #
    # case_names = []
    # event_names = []
    # timestamp = []
    # p_event = []
    # current_real = []
    #
    # real_diff = []
    # pred_diff = []
    #
    # for i in log.keys():
    #     for x in range(len(log[i][0])):
    #         case_names.append(i)
    #         event_names.append(log[i][0][x])
    #         timestamp.append(log[i][1][x])
    #         p_event.append(log[i][6][x])
    #         current_real.append(log[i][7][x])
    #
    #         real_diff.append(log[i][5][x])
    #         pred_diff.append(log[i][8][x])
    #
    # real_diff.append(0)
    #
    # frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
    #               'TimeStamp': timestamp, 'Current_Event': current_real, 'Predicted_Event': p_event,
    #               'Real_Diff': real_diff[1:], 'Predicted_Diff': pred_diff}
    # CHANGE THIS INTO TEST FILE PREDICTION WHEN DONE

    ### TEST

    data_test['event time:timestamp'] = pd.to_datetime(data_test['event time:timestamp'])

    # Convert from string to datetime
    data_test['event time:timestamp'] = pd.to_datetime(data_test['event time:timestamp'])

    data_test = data_test.sort_values(by=['case concept:name', 'event time:timestamp'])

    data_test['day_of_week'] = data_test['event time:timestamp'].dt.dayofweek

    data_test.to_csv("fixed_test.csv")

    t_log = dict()

    with open('fixed_test.csv', 'r') as file_test:
        next(file_test)
        for line in file_test:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')

            caseid = parts[2]

            task = parts[3]
            timestamp = parts[5]
            day = parts[6]

            if caseid not in t_log:
                t_log[caseid] = [[], [], []]

            t_log[caseid][0].append(task)
            t_log[caseid][1].append(timestamp)
            t_log[caseid][2].append(day)
    file.close()

    bugs = []

    for i in t_log.keys():
        if len(t_log[i][0]) == 1:
            bugs.append(i)

    for x in bugs:
        del t_log[x]

    #print(t_log)

    pbar = ProgressBar()

    for i in pbar(t_log.keys()):
        ID = []
        stamps = []
        for pairID, stamp in zip(pairwise(t_log[i][0]), pairwise(t_log[i][1])):
            ID.append(pairID)
            stamps.append(stamp)

        t_log[i].append(ID)
        t_log[i].append(stamps)

    pbar = ProgressBar()

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

    for i in pbar(t_log.keys()):
        # Add the real time differences
        real_diff = []
        for t in t_log[i][4]:
            real_diff.append(timeDiff(t))
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
                  'TimeStamp': timestamp, 'Current_Event': current_real, 'Predicted_Event': p_event,
                  'Real_Diff': real_diff[1:], 'Predicted_Diff': pred_diff}

    predicted_df = pd.DataFrame.from_dict(frame_dict)
    predicted_df.head(10)
    predicted_df.to_csv(out_path)

    os.remove('./fixed.csv')
    os.remove('./fixed_test.csv')
    return print("Training Time --- %s seconds ---" % (time.time() - start_time))


main()