import pandas as pd
import numpy as np
dfmean = pd.DataFrame()
df1 = pd.read_csv('E:\\zhaozhibo\\07 GCN\\09 数据处理\\07 日期对齐\\432_1800_cs.csv')
for i in range (4,df1.shape[1]):
    data1 = df1.iloc[:,i]
    max_value = np.max(data1)
    min_value = np.min(data1)
    range_value = max_value-min_value
    mean_value = np.mean(data1)
    data1 = (data1-min_value)/range_value
    dfmean = pd.concat([dfmean,data1],axis=1)
dfmean.to_csv("E:\\zhaozhibo\\07 GCN\\09 数据处理\\07 日期对齐\\参数归一化.csv")
