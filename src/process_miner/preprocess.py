import pandas as pd

def pre_data(train_path: str, test_path: str) -> object:

    if train_path == './data/2018-train.csv':
        df2_test = pd.read_csv(test_path, encoding = "ISO-8859-1",
        error_bad_lines = False, dtype = {'event org:resource': str}, engine='python')
        df2_train = pd.read_csv(train_path, encoding = "ISO-8859-1",
        error_bad_lines = False, dtype = {'event org:resource': str}, engine='python')
    else:
        df2_test = pd.read_csv(train_path)
        df2_train = pd.read_csv(test_path)

    # Convert from string to datetime
    df2_train['event time:timestamp'] = pd.to_datetime(df2_train['event time:timestamp'])
    df2_test['event time:timestamp'] = pd.to_datetime(df2_test['event time:timestamp'])

    # Concatenate the datasets
    df2 = pd.concat([df2_train, df2_test])

    # Sort the values and reset the index accordingly
    df2 = df2.sort_values(by=['event time:timestamp'])
    df2 = df2.reset_index(drop=True)

    # Create dataframe with first events of each case in chronological order
    df_cases_first_event = df2.drop_duplicates(subset='case concept:name', keep='first')
    df_cases_first_event = df_cases_first_event.reset_index(drop=True)

    # Split the cases in 80-20
    index80 = int(len(df_cases_first_event) * 80 / 100 - 1)
    df_cases_train = df_cases_first_event.loc[:index80]
    df_cases_test = df_cases_first_event.loc[index80 + 1:]

    # Training and test set complete
    df_train_new = df2.loc[df2['case concept:name'].isin(list(df_cases_train['case concept:name']))]
    df_test_new = df2.loc[df2['case concept:name'].isin(list(df_cases_test['case concept:name']))]

    # Setting the 'now' (as the last event of the last case in the training set)
    # now = df_train_new[df_train_new['case concept:name']==df_cases_train.iloc[-1]\
    #    ['case concept:name']].iloc[-1]['event time:timestamp']
    if 'case LoanGoal' in df2_train:
        now = pd.Timestamp('2016-12-08')
    elif len(df2_train.columns) == 5:
        now = pd.Timestamp('2012-12-12')
    elif 'case AMOUNT_REQ' in df2_train:
        now = pd.Timestamp('2012-10-02')
    else:
        now = pd.Timestamp('2017-11-12')

    # Training and test set before 'now'
    df_train_new_now = df_train_new.loc[(df_train_new['event time:timestamp'] <= now)]
    df_test_new_now = df_test_new.loc[(df_test_new['event time:timestamp'] <= now)]

    return df_train_new_now, df_test_new_now