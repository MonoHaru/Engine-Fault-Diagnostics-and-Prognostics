import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import math

def trend(time, slope = 0):
    return time * slope

def time_series_dataframe_ML():
    path_temp_gps = './temp_add_gps/'
    list_temp_gps = os.listdir(path_temp_gps)

    m1 = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[0]))
    for i in range(1,8):
        tmp = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[i]))
        m1 = pd.concat([m1, tmp], axis = 0)

    m1 = m1.reset_index(drop = True)
    m1 = m1[m1['TEMP']>=243.07].reset_index(drop = True)
    time_df = m1
    time_df = time_df.loc[:, ['TEMP']]

    for i in range(1,8):
        globals()['df_'+str(i)+'_temp'] = time_df[60436*(i-1):60436*i].reset_index(drop = True)

    N = 6
    dx = (600 - df_1_temp['TEMP'].mean()) / N # ??? ??????? ???? ?????? : 56.3785
    dx_minute = dx / (len(df_1_temp)-1) # ???? ??????

    time = np.arange(len(df_1_temp))
    slope = dx_minute * 2



    for i in range(1,8):
        mean = globals()['df_'+str(i)+'_temp']['TEMP'].mean()
        diff  = 280.40784269309677 - mean
        globals()['df_'+str(i)+'_temp']['TEMP'] += diff

    for i in range(8,13): # 8, 9, 10, 11, 12
        globals()['df_'+str(i)+'_temp'] = globals()['df_'+str(i-7)+'_temp'].copy()

    for i in range(2,13):
        series = np.round(trend(time, slope = slope) + globals()['df_'+str(i)+'_temp']['TEMP'] + dx*(i-2), 3)
        globals()['df_'+str(i)+'_temp']['TEMP'] = series

    temp_len = len(df_1_temp) * 12
    label = np.array([[i] * 120872 for i in range(1,7)]).reshape(-1)
    temp_TIME = pd.DataFrame({'TIME' : np.arange(temp_len)})
    temp_label = pd.DataFrame({'label' : label})
    df_temp_all = pd.concat([df_1_temp, df_2_temp, df_3_temp, df_4_temp, df_5_temp, df_6_temp, df_7_temp, df_8_temp, df_9_temp, df_10_temp, df_11_temp, df_12_temp], axis = 0)
    df_temp_all = df_temp_all.reset_index(drop = True)
    df_temp_all = pd.concat([df_temp_all,temp_label, temp_TIME], axis = 1)
    
    return df_temp_all


def make_dataframe(window_size, stride):
    path_temp_gps = './temp_add_gps/'
    list_temp_gps = os.listdir(path_temp_gps)

    A = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[0]))
    for i in range(1,8):
        tmp = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[i]))
        A = pd.concat([A,tmp], axis = 0)

    A = A.reset_index()
    A.drop(columns=['index','Unnamed: 0'], inplace=True)
    B = A[A['TEMP']>=243.07].reset_index()
    B.drop(columns = ['index'], inplace = True)
    TIME = pd.DataFrame({'TIME' : np.arange(len(B))})
    df= pd.concat([B, TIME], axis = 1)
    
    for i in range(1,8):
        globals()['df_'+str(i)+'_tmp'] = df[60436*(i-1):60436*i].reset_index().drop(columns=['index'], axis=0)

    N = 6
    dx = (600 - df_1_tmp['TEMP'].mean()) / N # ??? ??????? ???? ?????? : 56.3785
    dx_minute = dx / (len(df_1_tmp)-1) # ???? ??????

    time = np.arange(len(df_1_tmp))
    slope = dx_minute * 2


    for i in range(1,8):
        mean = globals()['df_'+str(i)+'_tmp']['TEMP'].mean()
        diff  = 280.40784269309677 - mean
        globals()['df_'+str(i)+'_tmp']['TEMP'] += diff

    for i in range(8,13):
        globals()['df_'+str(i)+'_tmp'] = globals()['df_'+str(i-5)+'_tmp'].copy()

    for i in range(2,13):
        series = np.round(trend(time, slope = slope) + globals()['df_'+str(i)+'_tmp']['TEMP'] + dx*(i-2), 3)
        globals()['df_'+str(i)+'_tmp']['TEMP'] = series

    for i in range(1,13):
        data = {'MEAN_TEMP' : [], 'STD' : [], 'MIN' : [], 'MAX' : [], 'SKEW' : [], 'KURT' : [], 'MEDIAN':[], '25%' : [], '75%' : [], 'label' : []}

        for j in range(0,len(df_1_tmp),stride):
            temp = globals()['df_'+str(i)+'_tmp']['TEMP'][j:j+window_size]
            MEAN = np.round(np.mean(temp), 3)
            MIN = np.min(temp)
            MAX = np.max(temp)
            STD = np.std(temp)
            median = temp.median()
            skew = temp.skew()
            kurt = temp.kurt()
            a, b = np.percentile(temp, q = [25,75])

            data['label'].append(math.ceil(i/2))
            data['MEAN_TEMP'].append(MEAN)
            data['MIN'].append(MIN)
            data['MAX'].append(MAX)
            data['STD'].append(STD)
            data['SKEW'].append(skew)
            data['KURT'].append(kurt)
            data['MEDIAN'].append(np.round(median,3))
            data['25%'].append(np.round(a,3))
            data['75%'].append(np.round(b,3))

        globals()['group_'+str(i)] = pd.DataFrame(data)


    tmp = pd.concat([group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10, group_11, group_12], axis = 0)
    tmp = tmp.reset_index(drop = True)
    tmp = tmp.dropna(axis = 0).reset_index(drop = True)

    return tmp

rs = 96

def ML():
    df_ML = make_dataframe(60,1)
    X = df_ML.iloc[:, :9].values
    y = df_ML['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = rs, shuffle = True)


    RFC = RandomForestClassifier(n_estimators=100, max_depth = 50, random_state = rs)
    RFC.fit(X_train, y_train)
    RFC_predict = RFC.predict(X_test)
    RFC_ac = accuracy_score(y_test, RFC_predict)

    return RFC, RFC_ac