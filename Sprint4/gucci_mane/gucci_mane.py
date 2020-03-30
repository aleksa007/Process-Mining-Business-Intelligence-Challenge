import time, argparse

from pm_models import baseline, combs_algo, de_tree
from pre_ev import evaluate_acc_rmse, pre_data
import sys


def main():
    # Taking the arguments
    parser = argparse.ArgumentParser("PM_tool")
    parser.add_argument("train_file", help="Process mining model will be trained on this", type=str)
    parser.add_argument("test_file", help="Process mining model will be tested on this", type=str)
    parser.add_argument("output_file", help="Predictions will be saved into a new CSV with this name", type=str)
    parser.add_argument("--demo", help="Demo version, halved data", action='store_true')
    args = parser.parse_args()

    train_path = args.train_file
    test_path = args.test_file


    out_path = args.output_file

    # Start
    start_time = time.time()

    # Preprocess calls
    if train_path == './data/road-train.csv' and test_path == './data/road-test.csv':
        df_train, df_test = pre_data(train_path, test_path)
    elif train_path == './data/2017-train.csv' and test_path == './data/2017-test.csv':
        df_train, df_test = pre_data(train_path, test_path)
    elif train_path == './data/2018-train.csv' and test_path == './data/2018-test.csv':
        df_train, df_test = pre_data(train_path, test_path)
    elif train_path.startswith('./data/road') and test_path.startswith('./data/road'):
        df_train, df_test = pre_data(train_path, test_path)
    else:
        print("Oops, can't find those files.")
        sys.exit()
    # Baseline call
    base = baseline(df_train, df_test)

    # Combination algorithm call
    base_perm = combs_algo(df_train, df_test)

    # Adding baseline predictions
    ones = list(set(base['Case_ID']) - set(base_perm['Case_ID']))

    base_pred_ev = base['Baseline_Event']
    base_pred_ts = base['Baseline_TimeDiff']

    for case in ones:
        row = base[base['Case_ID'] == case].index
        base = base.drop(row)

    base_perm['Baseline_Event'] = base_pred_ev
    base_perm['Baseline_TimeDiff'] = base_pred_ts


    # Decision tree call
    base_perm_tree = de_tree(df_train, df_test)

    base_perm_tree['Combs_Event'] = base_perm['Combs_Event']
    base_perm_tree['Combs_TimeDiff'] = base_perm['Combs_TimeDiff']

    base_perm_tree['Baseline_Event'] = base_pred_ev
    base_perm_tree['Baseline_TimeDiff'] = base_pred_ts

    # Output
    #base_perm.to_csv(out_path, index = False)
    base_perm_tree.to_csv(out_path, index = False)

    # Evaluation call
    #evaluate_acc_rmse(base, base_perm)
    evaluate_acc_rmse(base, base_perm, base_perm_tree)
    # The End
    return print('Tool running time: {}'.format(time.time() - start_time))

# Once ready, delete below
#main()

if __name__ == '__main__':

    main()
    #os.system('python gucci_mane.py ./data/road-train.csv ./data/road-test.csv ./predict.csv')