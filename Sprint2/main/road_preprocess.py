import pandas as pd

def pre(train_path, test_path):

    df2_test = pd.read_csv(train_path)
    df2_train = pd.read_csv(test_path)

    #Convert from string to datetime
    df2_train['event time:timestamp'] = pd.to_datetime(df2_train['event time:timestamp'])
    df2_test['event time:timestamp'] = pd.to_datetime(df2_test['event time:timestamp'])

    #Concatenate the datasets
    df2 = pd.concat([df2_train, df2_test])

    #Sort the values and reset the index accordingly
    df2 = df2.sort_values(by=['event time:timestamp'])
    df2 = df2.reset_index(drop=True)


    #Create dataframe with first events of each case in chronological order
    df_cases_first_event = df2.drop_duplicates(subset='case concept:name', keep='first')
    df_cases_first_event = df_cases_first_event.reset_index(drop=True)

    #Split the cases in 80-20
    index80 = int(len(df_cases_first_event)*80/100-1)
    df_cases_train = df_cases_first_event.loc[:index80]
    df_cases_test = df_cases_first_event.loc[index80+1:]

    #Training and test set complete
    df_train_new = df2.loc[df2['case concept:name'].isin(list(df_cases_train['case concept:name']))]
    df_test_new = df2.loc[df2['case concept:name'].isin(list(df_cases_test['case concept:name']))]

    #Setting the 'now' (as the last event of the last case in the training set)
    #now = df_train_new[df_train_new['case concept:name']==df_cases_train.iloc[-1]\
    #    ['case concept:name']].iloc[-1]['event time:timestamp']

    now = pd.Timestamp('2012-12-12')


    #Training and test set before 'now'
    df_train_new_now = df_train_new.loc[(df_train_new['event time:timestamp'] <= now)]
    df_test_new_now = df_test_new.loc[(df_test_new['event time:timestamp'] <= now)]

    df_test_new_now.to_csv('./data/road-test-pre.csv', index = False)
    df_train_new_now.to_csv('./data/road-train-pre.csv', index = False)

    return print("Preprocessing of road data done.")

if __name__ == '__main__':
    pre('./data/road-train.csv', './data/road-test.csv')