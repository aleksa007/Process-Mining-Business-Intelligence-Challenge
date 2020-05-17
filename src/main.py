import time, argparse, sys, os

from process_miner.preprocess import pre_data
from process_miner.models import baseline, combs_algo, de_tree
from process_miner.evaluate import evaluate_acc_rmse

# from memory_profiler import profile

# @profile


def main():
    # Taking the arguments
    parser = argparse.ArgumentParser("PM_tool")
    parser.add_argument("train_file", help="Process mining model will be trained on this", type=str)
    parser.add_argument("test_file", help="Process mining model will be tested on this", type=str)
    parser.add_argument("output_file", help="Predictions will be saved into a new CSV with this name", type=str)

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

        if not os.path.exists('./build'):
            os.mkdir('./build')
            
    else:
        print("Oops, can't find those files.")
        sys.exit()

    # Baseline call
    base = baseline(df_train, df_test)

    # Combination algorithm call
    comb = combs_algo(df_train, df_test)

    # Decision tree call
    tree = de_tree(df_train, df_test)

    # Adding baseline predictions
    common_cases = list(set(base['Case_ID']) - set(comb['Case_ID']))  # only take the common case_id's between the
                                                                      # baseline and the combinations algorithm

    for case in common_cases:
        row = base[base['Case_ID'] == case].index
        base = base.drop(row)

    # [base = base.drop(base[base['Case_ID']  == case].index) for case in common_cases]

    # Appending predictions of the baseline, and Combinations algo to the DecisionTree dataframe
    tree['Combs_Event'] = comb['Combs_Event']
    tree['Combs_TimeDiff'] = comb['Combs_TimeDiff']

    tree['Baseline_Event'] = base['Baseline_Event']
    tree['Baseline_TimeDiff'] = base['Baseline_TimeDiff']

    outputfile = tree.copy()
    # drop the modified Actual Time difference column
    outputfile = outputfile.drop(['DTree_Actual_TimeDiff'], axis=1)

    # Output the csv file
    outputfile.to_csv(out_path, index=False)

    # Evaluation call
    evaluate_acc_rmse(base, comb, tree)

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
        print("The tool has finished. ")
