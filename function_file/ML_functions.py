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
    dx = (600 - df_1_temp['TEMP'].mean()) / N # ��ü �����Ϳ� ���� ������ : 56.3785
    dx_minute = dx / (len(df_1_temp)-1) # �д� ������

    time = np.arange(len(df_1_temp))
    slope = dx_minute * 2



    for i in range(1,8):
        mean = globals()['df_'+str(i)+'_temp']['TEMP'].mean()
        diff  = 280.40784269309677 - mean
        globals()['df_'+str(i)+'_temp']['TEMP'] += diff

    for i in range(8,12):
        globals()['df_'+str(i)+'_temp'] = globals()['df_'+str(i-5)+'_temp'].copy()

    for i in range(2,12):
        series = np.round(trend(time, slope = slope) + globals()['df_'+str(i)+'_temp']['TEMP'] + dx*(i-2), 3)
        globals()['df_'+str(i)+'_temp']['TEMP'] = series

    temp_len = len(df_1_temp) * 11
    label = np.array([[i] * 60436 for i in range(1,12)]).reshape(-1)
    temp_TIME = pd.DataFrame({'TIME' : np.arange(temp_len)})
    temp_label = pd.DataFrame({'label' : label})
    df_temp_all = pd.concat([df_1_temp, df_2_temp, df_3_temp, df_4_temp, df_5_temp, df_6_temp, df_7_temp, df_8_temp, df_9_temp, df_10_temp,df_11_temp], axis = 0)
    df_temp_all = df_temp_all.reset_index(drop = True)
    df_temp_all = pd.concat([df_temp_all,temp_label, temp_TIME], axis = 1)
    return df_temp_all, temp_TIME, temp_label




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

    for i in range(1,12):
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

            data['label'].append(i)
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


    tmp = pd.concat([group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10, group_11], axis = 0)
    tmp = tmp.reset_index(drop = True)
    tmp = tmp.dropna(axis = 0).reset_index(drop = True)


    temp_len = len(df_1_tmp) * 11
    temp_TIME = pd.DataFrame({'TIME' : np.arange(temp_len)})
    df_temp_all = pd.concat([df_1_tmp, df_2_tmp, df_3_tmp, df_4_tmp, df_5_tmp, df_6_tmp, df_7_tmp, df_8_tmp, df_9_tmp, df_10_tmp,df_11_tmp], axis = 0)
    df_temp_all = df_temp_all.reset_index(drop = True).drop(columns = ['date','kst','longitude','latitude','TIME'])
    df_temp_all = pd.concat([df_temp_all, temp_TIME], axis = 1)
    return tmp, df_temp_all






def ML(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    kn_ac = accuracy_score(y_test, knn_pred)

    lgbm = LGBMClassifier(n_estimators = 100)
    lgbm.fit(X_train, y_train,
            eval_metric = 'multi_logloss',
            eval_set = [(X_test, y_test)])
    lgbm_predict = lgbm.predict(X_test)
    lgbm_ac = accuracy_score(y_test, lgbm_predict)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc_pred = dtc.predict(X_test)
    dtc_ac = accuracy_score(y_test, dtc_pred)

    clf = RandomForestClassifier(n_estimators=100, max_depth = 50)
    clf.fit(X_train, y_train)
    clf_predict = clf.predict(X_test)
    clf_ac = accuracy_score(y_test, clf_predict)

    return kn_ac, lgbm_ac, dtc_ac, clf_ac

def THRESHOLD(data):
    if data>=797.314:
        return 11
    elif data>=740.9375:
        return 10
    elif data>=684.5565:
        return 9
    elif data>=628.1795:
        return 8
    elif data>=571.7965:
        return 7
    elif data>=515.4215:
        return 6
    elif data>=459.045:
        return 5
    elif data>=402.6645:
        return 4
    elif data>=346.28:
        return 3
    elif data>=289/9025:
        return 2
    else:
        return 1