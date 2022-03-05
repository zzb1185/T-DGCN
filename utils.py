# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import pandas as pd

def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj
    
def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L) 
    
def calculate_laplacian(adj, lambda_max=1):  
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)
    
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial,name=name)

def huanyuan(test1,real1,path,pre_len):
#test处理
    data_test1 = pd.DataFrame()
    a = test1.shape[0]-pre_len
    num = a / 354
    for j in range(432):
        data1 = test1.iloc[:, j]
        ser1 = []
        for i in range(0, 354):
            a = data1[i * num]
            b = data1[i * num + 1]
            mean = (a + b) / 2
            ser1.append(mean)
        data_one1 = pd.DataFrame(ser1)
        data_test1 = pd.concat([data_test1, data_one1], axis=1)
    data_test1.to_csv(path + '/test.csv', encoding='utf-8')
#real处理
    data_real = pd.DataFrame()
    a = data1.shape[0]
    num = a / 354
    for j in range(432):
        data1 = real1.iloc[:, j]
        ser = []
        for i in range(0, 354):
            a = data1[i * num]
            b = data1[i * num + 1]
            mean = (a + b) / 2
            ser.append(mean)
        data_one = pd.DataFrame(ser)
        data_real = pd.concat([data_real, data_one], axis=1)
    data_real.to_csv(path + '/real.csv', encoding='utf-8')
    return data_test1,data_real