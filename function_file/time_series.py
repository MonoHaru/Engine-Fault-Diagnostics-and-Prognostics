import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

def trend(time, slope = 0):
    return time * slope

def time_series_dataframe():
    path_temp_gps = './temperature_csv_file/'
    list_temp_gps = os.listdir(path_temp_gps)

    A = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[0]))
    for i in range(1,8):
        tmp = pd.read_csv(os.path.join(path_temp_gps + list_temp_gps[i]))
        A = pd.concat([A,tmp], axis = 0)

    A = A.reset_index(drop = True)
    
    df = A[A['TEMP']>=243.07].reset_index(drop = True)
    df = df.drop(columns = ['date','kst'])

    


    for i in range(1,8):
        globals()['df_'+str(i)+'_tmp'] = df[60436*(i-1):60436*i].reset_index().drop(columns=['index'], axis=0)

    N = 6
    dx = (600 - df_1_tmp['TEMP'].mean()) / N # ��ü �����Ϳ� ���� ������ : 56.3785
    dx_minute = dx / (len(df_1_tmp)-1) # �д� ������

    time = np.arange(len(df_1_tmp))
    slope = dx_minute * 2

    for i in range(1,8):
        mean = globals()['df_'+str(i)+'_tmp']['TEMP'].mean()
        diff  = 280.40784269309677 - mean
        globals()['df_'+str(i)+'_tmp']['TEMP'] += diff

    for i in range(8,12):
        globals()['df_'+str(i)+'_tmp'] = globals()['df_'+str(i-5)+'_tmp'].copy()

    for i in range(2,12):
        series = np.round(trend(time, slope = slope) + globals()['df_'+str(i)+'_tmp']['TEMP'] + dx*(i-2), 3)
        globals()['df_'+str(i)+'_tmp']['TEMP'] = series

    df_temp_all = pd.concat([df_1_tmp, df_2_tmp, df_3_tmp, df_4_tmp, df_5_tmp, df_6_tmp, df_7_tmp, df_8_tmp, df_9_tmp, df_10_tmp,df_11_tmp], axis = 0)

    #df_temp_all = pd.concat([df_1_tmp, df_4_tmp, df_9_tmp, df_1_tmp, df_3_tmp, df_10_tmp, df_1_tmp, df_2_tmp, df_9_tmp, df_10_tmp, df_1_tmp,df_5_tmp, df_2_tmp, df_7_tmp, df_9_tmp ], axis = 0)
    df_temp_all = df_temp_all.reset_index(drop = True)
    temp_len = len(df_1_tmp) * 11
    temp_TIME = pd.DataFrame({'TIME' : np.arange(temp_len)})
    #label = np.array([[int(i)] * 60436 for i in [1,4,9,1,3,10,1,2,9,10,1,5,2,7,9]]).reshape(-1)
    label = np.array([[i] * 60436 for i in range(1,12)]).reshape(-1)
    temp_label = pd.DataFrame({'label' : label})
    #temp_label = pd.DataFrame({'label' : label})

    df_temp_all = pd.concat([df_temp_all, temp_TIME, temp_label], axis = 1)
    
    return df_temp_all
