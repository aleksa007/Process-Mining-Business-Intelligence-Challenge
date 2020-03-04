import pandas as pd

def pre(train_path, test_path):
    #Loading the training and test datasets
    # df2_test = pd.read_csv('./data/road-test.csv')
    # df2_train = pd.read_csv('./data/road-train.csv')

    df2_test = pd.read_csv(train_path)
    df2_train = pd.read_csv(test_path)


    #Change names of the columns
    df2_test = df2_test.rename(columns={"eventID ":"eventID",
                                      "case concept:name": "case_concept_name",
                                      "event concept:name": "event_concept_name",
                                      "event lifecycle:transition":"event_lifecycle_transition",
                                      "event time:timestamp":"event_time_timestamp"})

    df2_train = df2_train.rename(columns={"eventID ":"eventID",
                                        "case concept:name": "case_concept_name",
                                        "event concept:name": "event_concept_name",
                                        "event lifecycle:transition":"event_lifecycle_transition",
                                        "event time:timestamp":"event_time_timestamp"})

    #Convert from string to datetime
    df2_train['event_time_timestamp'] = pd.to_datetime(df2_train['event_time_timestamp'])
    df2_test['event_time_timestamp'] = pd.to_datetime(df2_test['event_time_timestamp'])


    #Concatenate the datasets
    df2 = pd.concat([df2_train, df2_test])

    #Sort the values and reset the index accordingly
    df2 = df2.sort_values(by=['event_time_timestamp'])
    df2 = df2.reset_index(drop=True)


    #Create dataframe with first events of each case in chronological order
    df_cases_first_event = df2.drop_duplicates(subset='case_concept_name', keep='first')
    df_cases_first_event = df_cases_first_event.reset_index(drop=True)

    #Split the cases in 80-20
    index80 = int(len(df_cases_first_event)*80/100-1)
    df_cases_train = df_cases_first_event.loc[:index80]
    df_cases_test = df_cases_first_event.loc[index80+1:]

    #Training and test set complete
    df_train_new = df2.loc[df2['case_concept_name'].isin(list(df_cases_train['case_concept_name']))]
    df_test_new = df2.loc[df2['case_concept_name'].isin(list(df_cases_test['case_concept_name']))]

    #Setting the 'now' (as the last event of the last case in the training set)
    #now = df_train_new[df_train_new['case_concept_name']==df_cases_train.iloc[-1]\
    #    ['case_concept_name']].iloc[-1]['event_time_timestamp']

    now = pd.Timestamp('2012-12-12')


    #Training and test set before 'now'
    df_train_new_now = df_train_new.loc[(df_train_new['event_time_timestamp'] <= now)]
    df_test_new_now = df_test_new.loc[(df_test_new['event_time_timestamp'] <= now)]

    df_test_new_now.to_csv('./data/road-test-pre.csv', index = False)
    df_train_new_now.to_csv('./data/road-train-pre.csv', index = False)

    return "Preprocessing of road data done."

if '__name__' == '__main__':
    pre('./data/road-train.csv', './data/road-test.csv')