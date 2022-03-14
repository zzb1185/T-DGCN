import math
import numpy as np
import pandas as pd


def dtw(file, num1):
    num = num1
    distmatrix = np.zeros((num, num))
    for i in range(num):
        t = file.iloc[:, i]
        for j in range(i + 1, num):
            r = file.iloc[:, j]
            # print(files[j])
            if len(r) == 0:
                print("Notdata：", file[j])
            distmatrix[i, j] = mydtw_function2(t, r)
        print("NO.{0}finish!".format(i))
    return distmatrix

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
            D1 = D[i - 1, j]
            if j > 0:
                D2 = D[i - 1, j - 1]
                D3 = D[i, j - 1]
            else:
                D2 = np.inf
                D3 = np.inf
            D[i, j] = d[i, j] + min([D1, D2, D3])
    dist = D[n - 1, m - 1]

    dist = math.exp(-dist)
    return dist


if __name__ == '__main__':
    dfall = pd.read_csv(r'data/testdata/Well_Details.csv')
    dfxy = dfall.iloc[:, 1:3]  # 取出经纬度计算
    # 01 计算邻近关系
    num = dfxy.shape[0]  # 多少口井
    dfnear = pd.DataFrame(columns=['one', 'two'])  # 临近矩阵
    for i in range(num):
        x1 = dfxy.iloc[i, 0]
        y1 = dfxy.iloc[i, 1]
        for j in range(num):
            x2 = dfxy.iloc[j, 0]
            y2 = dfxy.iloc[j, 1]
            dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (0.5)
            if dis <= 350:  #350m
                dfnear = dfnear.append(pd.DataFrame({'one': [i], 'two': [j]}))  # 临近矩阵
    print("邻近关系计算完成")

    # 02 归一化
    dfmean = pd.DataFrame()  # 归一化之后的df
    for i in range(3, dfall.shape[1]):
        data1 = dfall.iloc[:, i]
        max_value = np.max(data1)
        min_value = np.min(data1)
        range_value = max_value - min_value
        mean_value = np.mean(data1)
        data1 = (data1 - min_value) / range_value
        dfmean = pd.concat([dfmean, data1], axis=1)  # 参数归一化之后的df
    print("归一化完成")

    # 03 计算dtw距离

    dfmeant = pd.DataFrame(dfmean.T)
    num1 = dfmeant.shape[1]
    distmatrix = dtw(dfmeant, num1)
    dfmat = pd.DataFrame(distmatrix)  # 大矩阵
    print("DTW矩阵计算完成")

    # 04 组合
    num_finished = 1
    oh432 = np.zeros((num, num))
    for row in range(dfnear.shape[0]):
        input = dfnear.iloc[(row, 0)]
        near = dfnear.iloc[(row, 1)]
        qz1 = dfmat.iloc[(input, near)]
        qz2 = dfmat.iloc[(near, input)]
        if (input == near):
            qz = 1
        else:
            if qz1 > qz2:
                qz = qz1
            else:
                qz = qz2
        oh432[input][near] = qz
        print(row)

    dfohe = pd.DataFrame(oh432)  # 完全体权重邻接矩阵
    dfohe.to_csv(r"data\testdata\test_OHe.csv",index=False,header=None)
    print("矩阵组装完成，保存在data\\testdata中")

