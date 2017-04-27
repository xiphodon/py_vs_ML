#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/27 9:59
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : dictionary_learning.py
# @Software: PyCharm Community Edition


'''
数据预处理
（字典学习）
'''

from sklearn.decomposition import DictionaryLearning


def test_DictionaryLearning():
    '''
    测试 DictionaryLearning 的用法
    :return: None
    '''
    X = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [10, 9, 8, 7, 6, ],
         [5, 4, 3, 2, 1]]
    print("before transform:", X)
    dct = DictionaryLearning(n_components=3)
    dct.fit(X)
    print("components is :", dct.components_)
    print("after transform:", dct.transform(X))


if __name__ == '__main__':
    test_DictionaryLearning()  # 调用 test_DictionaryLearning
