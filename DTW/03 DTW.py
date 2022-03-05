import pandas as pd
import os
import numpy as np
import time
import openpyxl
import math


def mydtw_function2(t, r):
    n = len(t)
    m = len(r)
    t = np.array(t)
    r = np.array(r)
    d = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d[i, j] = np.sum((t[i] - r[j]) ** 2)

    # 累积距离
    D = np.ones((n, m)) * np.inf
    D[0, 0] = d[0, 0]
    # 动态规划
    for i in range(1, n):
        for j in range(m):
            D1 = D[i-1, j]
            if j > 0:
                D2 = D[i-1, j-1]
                D3 = D[i, j - 1]
            else:
                D2 = np.inf
                D3 = np.inf
            D[i, j] = d[i, j] + min([D1, D2, D3])
    dist = D[n-1, m-1]
    # dist = 2**(-dist)
    dist = math.exp(-dist)
    return dist


def dtw(file,num1):
    num = num1
    distmatrix = np.zeros((num, num))
    for i in range(num):
        # t = pd.read_excel(os.path.join(root_dir, files[i]), usecols=["砂比"])
        t = file.iloc[:,i]
        for j in range(i+1, num):
            # r = pd.read_excel(os.path.join(root_dir, files[j]), usecols=["砂比"])
            r = file.iloc[:,j]
            # print(files[j])
            if len(r) == 0:
                print("Notdata：", file[j])
            distmatrix[i, j] = mydtw_function2(t, r)
        print("NO.{0}finish!".format(i))
    return distmatrix


if __name__ == '__main__':
    time1 = time.time()
    file = pd.read_excel('E:\\zhaozhibo\\07 GCN\\09 数据处理\\07 日期对齐\\参数归一化.xlsx',sheet_name="Sheet1",header=None)
    num1 = file.shape[1]
    distmatrix = dtw(file,num1)
    pd.DataFrame(distmatrix).to_csv('E:\\zhaozhibo\\07 GCN\\09 数据处理\\07 日期对齐\\432_1800_DTWe.csv', index=False)