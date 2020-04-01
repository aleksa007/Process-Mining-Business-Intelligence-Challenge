import time, argparse
#from memory_profiler import profile
from pm_models import baseline, combs_algo, de_tree
from pre_ev import evaluate_acc_rmse, pre_data
import sys


#@profile
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
    datasets = ['./data/road-train.csv', './data/2017-train.csv', './data/2018-train.csv', './data/2012-train.csv']
    test_datasets = ['./data/road-test.csv', './data/2017-test.csv', './data/2018-test.csv', './data/2012-test.csv']
    # Preprocess calls
    if train_path in datasets and test_path in test_datasets:
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

    for case in ones:
        row = base[base['Case_ID'] == case].index
        base = base.drop(row)

    base_pred_ev = base['Baseline_Event']
    base_pred_ts = base['Baseline_TimeDiff']

    base_perm['Baseline_Event'] = base_pred_ev
    base_perm['Baseline_TimeDiff'] = base_pred_ts


    # Decision tree call
    base_perm_tree = de_tree(df_train, df_test)

    base_perm_tree['Combs_Event'] = base_perm['Combs_Event']
    base_perm_tree['Combs_TimeDiff'] = base_perm['Combs_TimeDiff']

    base_perm_tree['Baseline_Event'] = base_pred_ev
    base_perm_tree['Baseline_TimeDiff'] = base_pred_ts

    outputfile = base_perm_tree.copy()
    outputfile = outputfile.drop(['DTree_Actual_TimeDiff'], axis = 1)
    # Output
    outputfile.to_csv(out_path, index = False)

    # Evaluation call
    evaluate_acc_rmse(base, base_perm, base_perm_tree)

    # The End
    return print('Tool running time: {}'.format(time.time() - start_time))

if __name__ == '__main__':

    main()
    try:
        import winsound
        duration = 2000
        freq = 840
        winsound.Beep(freq, duration)
    except:
        print("Tool is done.")
