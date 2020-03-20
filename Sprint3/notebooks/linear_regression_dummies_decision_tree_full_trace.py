#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from math import sqrt
import datetime
from datetime import date
from itertools import repeat 
from progressbar import ProgressBar
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score
import time


# In[2]:


start_time = time.time()


# In[3]:


#Loading the datasets
os.chdir("../")
os.getcwd()


# In[4]:


data_train = pd.read_csv("./data/road-train-pre.csv")
data_test = pd.read_csv("./data/road-test-pre.csv")


# In[5]:


#Sort the datasets and trainsofm to datetime
data_train['event time:timestamp'] = pd.to_datetime(data_train['event time:timestamp'])
data_test['event time:timestamp'] = pd.to_datetime(data_test['event time:timestamp'])
data_train=data_train.sort_values(by=["case concept:name", "event time:timestamp"])
data_test=data_test.sort_values(by=["case concept:name", "event time:timestamp"])


# # Decision Tree

# Train Data

# In[6]:


data_train.to_csv("fixed.csv")

log = dict()
with open('fixed.csv', 'r') as file:
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
            log[caseid] = [[],[]]

        log[caseid][0].append(task)
        log[caseid][1].append(timestamp)
        
file.close()

os.remove('fixed.csv')


# In[7]:


for i in log.keys():
    current = log[i][0]  # recording the current case' events
    
    real_next = current[1:]  # next real event
    real_next.append('New Case')  # adding a 'new case' as real next event for every last event
    
    log[i].append(real_next)  # adding the real next events to the log file


# Test Data

# In[8]:


data_test.to_csv("fixed_test.csv")

log_test = dict()
with open('fixed_test.csv', 'r') as file:
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
            log_test[caseid] = [[],[]]

        log_test[caseid][0].append(task)
        log_test[caseid][1].append(timestamp)
        
file.close()

os.remove('fixed_test.csv')


# In[9]:


"""Fixing a bug of cases that are in the test data but are incomplete due to the train-test split."""

bugs = []

for i in log_test.keys():  #  recording the cases which have events cut because of the train - test split
    if len(log_test[i][0]) == 1:
        bugs.append(i)
            
for x in bugs:  # deleting the above mentioned events 
    del log_test[x]
    data_test.drop(data_test.index[data_test['case concept:name'] == x], inplace = True)


# In[10]:


for i in log_test.keys():
    current = log_test[i][0]  # current case' events
    
    real_next = current[1:]  # next real event
    real_next.append('New Case')  # adding a 'new case' as real next event for every last event
    log_test[i].append(real_next)


# In[11]:


m = 0

tenabove = []
for i in log.keys():
    if len(log[i][0]) > m:
        m = len(log[i][0])
        
    if len(log[i][0]) > 10:
        tenabove.append(i)
        
tenabove


# In[12]:


m_t = 0

tenabove_test = []
for i in log_test.keys():
    if len(log_test[i][0]) > m:
        m = len(log_test[i][0])
        
    if len(log_test[i][0]) > 10:
        tenabove_test.append(i)
        
tenabove_test


# In[13]:


delete = []
for i in log_test.keys():
    if len(log_test[i][0]) > m:
        print(i)
        print(log_test[i])
        delete.append(i)
        #m_t = len(log_test[i][0])
        
for i in delete:
    print(i)
    
    data_test.drop(data_test.index[data_test['case concept:name'] == i], inplace = True)

    del log_test[i]


# Store Data Train

# In[14]:


train_data = {} 

for i in log.keys():
    for x in log[i][0]:
        case = log[i][0]
        ind = log[i][0].index(x)
        
        if ind not in train_data:
            train_data[ind] = [[],[]]
        
        
        train_data[ind][0].append(case[:ind+1])  # appending the trace
        
        if ind < len(case)-1:
            train_data[ind][1].append(case[ind+1])  # appending the next event of the trace
            
        elif ind == len(case)-1:
            train_data[ind][1].append('New Case')


# Store Data Test

# In[15]:


test_data = {} 

for i in log_test.keys():
    for x in log_test[i][0]:
        case = log_test[i][0]
        ind = log_test[i][0].index(x)
        
        if ind not in test_data:
            test_data[ind] = [[],[]]
        
        
        test_data[ind][0].append(case[:ind+1])  # appending the trace
        
        if ind < len(case)-1:
            test_data[ind][1].append(case[ind+1])  # appending the next event of the trace
            
        elif ind == len(case)-1:
            test_data[ind][1].append('New Case')


# Encoding

# In[16]:


cases = list(data_train['event concept:name'].unique()) + list(data_test['event concept:name'].unique())
cases.append('New Case')
cases = list(set(cases))
le = preprocessing.LabelEncoder()
le.fit(cases)  # encoding all event names into integers


# In[17]:


pbar = ProgressBar()

for i in pbar(train_data.keys()):
    
    encoded = []
    for trace in train_data[i][0]:  # encoding all strings into integers in the trace
        local_encoded = []
        for event in trace:
            local_encoded.append(int(le.transform([event])))
        encoded.append(local_encoded)
    
    train_data[i][0] = np.array(encoded)
    
    
    encoded_next = []
    for g in train_data[i][1]:
        encoded_next.append(int(le.transform([g])))
                            
                            
    train_data[i][1] = np.array(encoded_next)


# In[18]:


pbar = ProgressBar()

for i in pbar(test_data.keys()):
    
    encoded = []
    for trace in test_data[i][0]:  # encoding all strings into integers in the trace
        local_encoded = []
        for event in trace:
            local_encoded.append(int(le.transform([event])))
        encoded.append(local_encoded)
    
    test_data[i][0] = np.array(encoded)
    
    
    encoded_next = []
    for g in test_data[i][1]:
        encoded_next.append(int(le.transform([g])))
                            
                            
    test_data[i][1] = np.array(encoded_next)


# Training

# In[19]:


def decision_tree(pos):

    x_train= train_data[pos][0]
    y_train= train_data[pos][1]

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    
    return classifier


# In[20]:


predictors = {}

for i in range(len(test_data)):
    if i >= len(train_data) - 1:
        predictors[i] = decision_tree(len(train_data) - 1)
        
    else:
        predictors[i] = decision_tree(i)


# Adding Predictions

# In[21]:


pbar = ProgressBar()

for i in pbar(log_test.keys()):
    current = log_test[i][0]
    
    
    encoded = []
    for g in current:
        encoded.append(int(le.transform([g])))
    encoded = np.array(encoded)
    log_test[i].append(encoded)


# In[22]:


pbar = ProgressBar()
for i in pbar(log_test.keys()):
    
    current_encoded = log_test[i][3]
    predictions = []
    
    for x in current_encoded:
        ind = list(current_encoded).index(x)
        
        if ind >= len(train_data) - 1:
            
            tree = predictors[len(train_data) - 1]
            p = current_encoded[:(len(train_data))]
            p = p.reshape(1, -1)
            pred = tree.predict(p)
            pred_string = le.inverse_transform(pred)[0]
            predictions.append(pred_string)
            
            
            
        else:
        
            tree = predictors[ind]
            p = current_encoded[:ind+1]
            p = p.reshape(1, -1)
            pred = tree.predict(p)
            pred_string = le.inverse_transform(pred)[0]
            predictions.append(pred_string)
        
    log_test[i].append(predictions)


# Evaluation

# In[23]:


case_names = []
event_names = []
timestamp = []
p_event = []
current_real = []

for i in log_test.keys():
    for x in range(len(log_test[i][0])):
        case_names.append(i)
        event_names.append(log_test[i][0][x])
        timestamp.append(log_test[i][1][x])
        p_event.append(log_test[i][4][x])
        current_real.append(log_test[i][2][x])


frame_dict = {'Case_ID': case_names, 'Event_Name': event_names,
              'TimeStamp': timestamp, 'Next_Event': current_real, 'Predicted_Event': p_event}
predicted_df = pd.DataFrame.from_dict(frame_dict)

event_real = np.array(predicted_df['Next_Event'])
event_pred = np.array(predicted_df['Predicted_Event'])

acc = accuracy_score(event_real, event_pred)
print('Accuracy for event prediction TEST SET: {}%'.format(round(acc, 2) * 100))


# In[24]:


predicted_df


# # Linear Regression 

# Train Data Preprocessing

# In[25]:


#Add new useful columns for the model train
data_train['position_event']=data_train.groupby('case concept:name').cumcount()
data_train['position_event']= data_train['position_event'] + 1
data_train['week_day']=data_train['event time:timestamp'].dt.dayofweek


# In[26]:


data_train


# In[27]:


#Encoding all event names into integers
cases = data_train['event concept:name'].unique().tolist()
cases.insert(0, 'New Case')
le_case = preprocessing.LabelEncoder()
le_case.fit(cases)


# In[28]:


#Encoding lifecycle into integers
life = data_train['event lifecycle:transition'].unique().tolist()
le_life = preprocessing.LabelEncoder()
le_life.fit(life)


# In[29]:


#Preprocess data for model train
#Event poistion
x_train_position = np.array(data_train['position_event']).reshape(-1, 1)[:]
#Previous event
x_train_prev = list(data_train['event concept:name'])
x_train_prev= le_case.transform(x_train_prev)
x_train_prev = np.array(x_train_prev).reshape(-1,1)[:]
# Event
x_train_event = list(data_train['event concept:name'])
x_train_event.insert(len(data_train), 'New Case')
x_train_event= le_case.transform(x_train_event)
x_train_event = np.array(x_train_event).reshape(-1,1)[1:]
#Day of the week previous event event
x_train_week = list(data_train['week_day'])
x_train_week = np.array(x_train_week).reshape(-1,1)[:]
#Timestamp event
data_train[['event time:timestamp']] = data_train[['event time:timestamp']].astype(str)
x_train_date = list(data_train['event time:timestamp'])
x_train_date.insert(len(data_train), None)
x_train_date=np.array(x_train_date).reshape(-1,1)[1:]
#Timestamp previous event
x_train_date_prev = list(data_train['event time:timestamp'])
x_train_date_prev=np.array(x_train_date_prev).reshape(-1,1)[:]
#Event Lifecycle
x_train_life = list(data_train['event lifecycle:transition'])
x_train_life= le_life.transform(x_train_life)
x_train_life = np.array(x_train_life).reshape(-1,1)[:]


# In[30]:


#Length case for train set
cases = data_train.groupby(['case concept:name'])
per_case = pd.DataFrame({'no of events':cases['eventID '].count()})
lst_per_case = per_case["no of events"].tolist()
case_length = []
for length in lst_per_case:
    case_length.extend(repeat(length, length))
x_train_length_case=np.array(case_length).reshape(-1,1)[:]


# In[31]:


#Combine features for the model train
x_train_new = np.concatenate((x_train_position,x_train_prev, x_train_event, x_train_week, x_train_date, x_train_date_prev, x_train_length_case, x_train_life), axis=1)


# In[32]:


#Add features to new dataframe train
df_train = pd.DataFrame(data=x_train_new, columns=['position_event', 'prev_event', 'event', 'week_day_prev', 'date', 'date_prev', 'case_length', 'lifecycle'])
df_train.loc[df_train['position_event'] == df_train['case_length'], 'event'] = 5
df_train[['date','date_prev']] = df_train[['date','date_prev']].apply(pd.to_datetime)
df_train.loc[df_train['event'] == 5, 'date'] = None
df_train['in_between'] = (df_train['date'] - df_train['date_prev']).dt.days
df_train.loc[df_train['event'] == 5, 'in_between'] = 0
df_train


# Train Dummies

# In[33]:


#Implementing dummies train
df_train=pd.get_dummies(df_train, columns=['event', 'prev_event', 'week_day_prev', 'position_event', 'lifecycle'])
df_train = df_train.drop(['date', 'date_prev'], 1)
df_train


# Test Data Preprocessing

# In[34]:


#Add new useful columns for the model test
data_test['position_event']=data_test.groupby('case concept:name').cumcount()
data_test['position_event']= data_test['position_event'] + 1
data_test['week_day']=data_test['event time:timestamp'].dt.dayofweek


# In[35]:


predicted_events=predicted_df['Predicted_Event'][:].tolist()
data_test['pred_event']=predicted_events


# In[36]:


len(predicted_events)


# In[37]:


#Preprocess data for model test
#Event poistion
x_test_position = np.array(data_test['position_event']).reshape(-1, 1)[:]
#Previous event
x_test_prev = data_test['event concept:name'].tolist()
x_test_prev = le_case.transform(x_test_prev)
x_test_prev = np.array(x_test_prev).reshape(-1,1)[:]
#Predicted Event
x_test_event = data_test['pred_event'].tolist()
x_test_event= le_case.transform(x_test_event)
x_test_event = np.array(x_test_event).reshape(-1,1)[:]
#Day of the week previous event
x_test_week = data_test['week_day'].tolist()
x_test_week = np.array(x_test_week).reshape(-1,1)[:]
#Timestamp event
data_test[['event time:timestamp']] = data_test[['event time:timestamp']].astype(str)
x_test_date = list(data_test['event time:timestamp'])
x_test_date.insert(len(data_test), None)
x_test_date=np.array(x_test_date).reshape(-1,1)[1:]
#Timestamp previous event
x_test_date_prev = list(data_test['event time:timestamp'])
x_test_date_prev=np.array(x_test_date_prev).reshape(-1,1)[:]
#Event Lifecycle
x_test_life = data_test['event lifecycle:transition'].tolist()
x_test_life= le_life.transform(x_test_life)
x_test_life = np.array(x_test_life).reshape(-1,1)[:]


# In[38]:


#Length case for test set
test_cases = data_test.groupby(['case concept:name'])
per_case_test = pd.DataFrame({'no of events':test_cases['eventID '].count()})
lst_per_case_test = per_case_test["no of events"].tolist()
case_length_test = []
for length in lst_per_case_test:
    case_length_test.extend(repeat(length, length))
x_test_length_case=np.array(case_length_test).reshape(-1,1)[:]


# In[39]:


#Combine features for the model test
x_test_new = np.concatenate((x_test_position ,x_test_prev, x_test_event, x_test_week, x_test_date, x_test_date_prev, x_test_length_case, x_test_life), axis=1)


# In[40]:


#Add features to new dataframe test
df_test = pd.DataFrame(data=x_test_new, columns=['position_event', 'prev_event', 'event', 'week_day_prev', 'date', 'date_prev', 'case_length', 'lifecycle'])
df_test.loc[df_test['position_event'] == df_test['case_length'], 'date'] = None
df_test[['date','date_prev']] = df_test[['date','date_prev']].apply(pd.to_datetime)
df_test['in_between'] = (df_test['date'] - df_test['date_prev']).dt.days
df_test.loc[df_test['position_event'] == df_test['case_length'], 'in_between'] = 0
df_test


# In[41]:


#Remove cases with more events than the cases in the train set
df_test=df_test[df_test['case_length']<=max(df_train['case_length'])]
df_test


# Test Dummies

# In[42]:


#Implementing dummies test
df_test=pd.get_dummies(df_test, columns=['event', 'prev_event', 'week_day_prev', 'position_event', 'lifecycle'])
df_test = df_test.drop(['date', 'date_prev'], 1)
df_test


# Feature selection and model training

# In[43]:


col_train=df_train.columns
col_test=df_test.columns
features=set(col_train).intersection(col_test)
features.discard('in_between')
X_train = df_train[features] # Features
y_train = df_train['in_between'] # Target variable
X_test = df_test[features] # Features
y_test = df_test['in_between'] # Target variable

#Training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[44]:


print(regressor.intercept_)
print(regressor.coef_)


# Evaluation

# In[45]:


y_pred = regressor.predict(X_test)
df_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_predictions


# In[46]:


#R-squared value for the model train
regressor.score(X_train, y_train)


# In[47]:


#R-squared value for the model test
regressor.score(X_test, y_test)


# In[48]:


#Root Mean Squared Error of the model
rmse = sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[49]:


print("--- %s seconds ---" % (time.time() - start_time))

