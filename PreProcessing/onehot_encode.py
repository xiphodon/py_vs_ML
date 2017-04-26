#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/26 16:30
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : onehot_encode.py
# @Software: PyCharm Community Edition

'''
数据预处理
（独热码）
'''

from sklearn.preprocessing import OneHotEncoder


def test_OneHotEncoder():
    '''
    测试 OneHotEncoder 的用法
    :return: None
    '''
    X = [[1, 2, 3, 4, 5],
         [5, 4, 3, 2, 1],
         [3, 3, 3, 3, 3, ],
         [1, 1, 1, 1, 1]]
    print("before transform:", X)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(X)
    print("active_features_:", encoder.active_features_)    #激活特征数组
    print("feature_indices_:", encoder.feature_indices_)    #原始特征拼接转换后特征的起始区间
    print("n_values_:", encoder.n_values_)  #原始属性的取值种类
    print("after transform:", encoder.transform([[1, 2, 3, 4, 5]])) #剔除后未激活特征，样本[1,2,3,4,5]的独热编码


if __name__ == '__main__':
    test_OneHotEncoder()  # 调用 test_OneHotEncoder
