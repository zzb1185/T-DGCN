import pandas as pd
import numpy as np
#没加权重的版本 
# oh533 = np.zeros((533,533))
# data3 = pd.read_csv('E:\\zhaozhibo\\07 GCN\\06 测试\\点距离\\300.csv')
# dfqz = pd.read_excel('属性距离值.xlsx')
# for row in range (data3.shape[0]):
#     input = data3.iloc[(row,1)]
#     near = data3.iloc[(row,2)]
#     qz1 = dfqz.iloc[(input,near)]
#     qz2 = dfqz.iloc[(near, input)]
#     if qz1>qz2:
#         qz = qz1
#     else:
#         qz = qz2
#     oh533[input][near] = qz
# x3 = pd.DataFrame(oh533)
# x3.to_csv('533权重onehot.csv')


oh533 = np.zeros((432,432))
path = "E:\zhaozhibo\\07 GCN\\09 数据处理\\08 三维点距\\"
# 临近表
data3 = pd.read_excel(path + '单向点距.xlsx',header=None)
# 权重表
dfqz = pd.read_csv(path+'432_1800_DTWe.csv',header=None)

for row in range (data3.shape[0]):
    input = data3.iloc[(row,0)]
    near = data3.iloc[(row,1)]
    qz1 = dfqz.iloc[(input,near)]
    qz2 = dfqz.iloc[(near, input)]
    if (input == near) :
        qz = 1
    else:
        if qz1>qz2:
            qz = qz1
        else:
            qz = qz2
    oh533[input][near] = qz
x3 = pd.DataFrame(oh533)
x3.to_csv(path+'432_1800_OHe_double_350.csv')