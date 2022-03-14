# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
import time

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    return trainX, trainY, testX, testY
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b)/la.norm(a)
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a - b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var


time_start = time.time()
path = r'data/432_1800_350/432_1800_cql_00001 - 副本.csv'
data = pd.read_csv(path)
data =np.mat(data,dtype=np.float32)
max_value = np.max(data)
min_value = np.min(data)
std_value = data.std()
mean_value = data.mean()
data = data/std_value


time_len = data.shape[0]
num_nodes = data.shape[1]
train_rate = 0.8
seq_len = 7
pre_len = 1
trainX,trainY,testX,testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
print("开始啦")
method = 'ARIMA' ####HA or SVR or ARIMA


########### HA #############
if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = testX[i]
        a1 = np.mean(a, axis=0) 
        result.append(a1)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1,num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1,num_nodes])
    rmse, mae, accuracy,r2,var = evaluation(testY1, result1)  
    print('HA_rmse:%r'%rmse,
          'HA_mae:%r'%mae,
          'HA_acc:%r'%accuracy,
          'HA_r2:%r'%r2,
          'HA_var:%r'%var)


############ SVR #############
if method == 'SVR':  
    total_rmse, total_mae, total_acc, result = [], [],[],[]
    for i in range(num_nodes):
        data1 = np.mat(data)
        #取第i列
        a = data1[:,i]
        # aX aY 是训练组的输入输出，分别为
        a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, pre_len])    
       
        svr_model=SVR(kernel ='rbf',coef0 = 0,gamma = 'scale',degree = 3,
                      tol = 0.001,C = 1.0,epsilon = 0.1,shrinking = True,
                      cache_size = 200,	verbose = False,max_iter = -1 )

        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len ,axis=1)
        result.append(pre)
        print(i)
    result1 = np.array(result)
    result1 = np.reshape(result1, [num_nodes,-1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY)


    testY1 = np.reshape(testY1, [-1,num_nodes])
    total = np.mat(total_acc)
    total[total<0] = 0
    testY1 = np.array(testY1)*std_value
    result1 = np.array(result1)*std_value
    testY11 = pd.DataFrame(testY1)
    result111 = pd.DataFrame(result1)
    path = "E:\\zhaozhibo\\07 GCN\\02 Code\\T-GCN\\out\\svr\\"+str(pre_len)
    #testY11.to_csv(path+'\\testY1.csv')
    #result111.to_csv(path+'\\result1.csv')
    rmse1, mae1, acc1,r2,var = evaluation(testY1, result1)
    print('SVR_rmse:%r'%rmse1,
          'SVR_mae:%r'%mae1,
          'SVR_acc:%r'%acc1,
          'SVR_r2:%r'%r2,
          'SVR_var:%r'%var)

######## ARIMA #########
if method == 'ARIMA':
    rng = pd.date_range('1/3/2012', periods=5664, freq='15min')
    a1 = pd.DatetimeIndex(rng)
    data.index = a1
    num = data.shape[1]   
    rmse,mae,acc,r2,var,pred,ori = [],[],[],[],[],[],[]
    for i in range(156):
        ts = data.iloc[:,i]
        ts_log=np.log(ts)    
        ts_log=np.array(ts_log,dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log,order=[1,0,0])
        properModel = model.fit()
        predict_ts = properModel.predict(4, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score)
        var.append(var_score)
#    for i in range(109,num):
#        ts = data.iloc[:,i]
#        ts_log=np.log(ts)    
#        ts_log=np.array(ts_log,dtype=np.float)
#        where_are_inf = np.isinf(ts_log)
#        ts_log[where_are_inf] = 0
#        ts_log = pd.Series(ts_log)
#        ts_log.index = a1
#        model = ARIMA(ts_log,order=[1,1,1])
#        properModel = model.fit(disp=-1, method='css')
#        predict_ts = properModel.predict(2, dynamic=True)
#        log_recover = np.exp(predict_ts)
#        ts = ts[log_recover.index]
#        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
#        rmse.append(er_rmse)
#        mae.append(er_mae)
#        acc.append(er_acc)  
#        r2.append(r2_score)
#        var.append(var_score)
    acc1 = np.mat(acc)
    acc1[acc1 < 0] = 0
    print('arima_rmse:%r'%(np.mean(rmse)),
          'arima_mae:%r'%(np.mean(mae)),
          'arima_acc:%r'%(np.mean(acc1)),
          'arima_r2:%r'%(np.mean(r2)),
          'arima_var:%r'%(np.mean(var)))
time_end = time.time()
print(time_end-time_start,'s')
