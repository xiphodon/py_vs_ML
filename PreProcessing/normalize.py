#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/26 17:00
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : normalize.py
# @Software: PyCharm Community Edition

'''
数据预处理
（数据正则化）
'''

from sklearn.preprocessing import Normalizer


def test_Normalizer():
    '''
    测试 Normalizer 的用法
    :return: None
    '''
    X = [[1, 2, 3, 4, 5],
         [5, 4, 3, 2, 1],
         [1, 3, 5, 2, 4, ],
         [2, 4, 1, 3, 5]]

    print("before transform:", X)
    normalizer = Normalizer(norm='l2')
    print("after transform:", normalizer.transform(X))


if __name__ == '__main__':
    test_Normalizer()  # 调用 test_Normalizer
