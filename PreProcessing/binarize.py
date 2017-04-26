#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/26 16:27
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : binarize.py
# @Software: PyCharm Community Edition

'''
数据预处理
（二元化）
'''

from sklearn.preprocessing import Binarizer


def test_Binarizer():
    '''
    测试 Binarizer 的用法
    :return: None
    '''
    X = [[1, 2, 3, 4, 5],
         [5, 4, 3, 2, 1],
         [3, 3, 3, 3, 3, ],
         [1, 1, 1, 1, 1]]
    print("before transform:", X)
    binarizer = Binarizer(threshold=2.5)    #threshold：阈值设定，高于阈值为1，低于阈值为0
    print("after transform:", binarizer.transform(X))


if __name__ == '__main__':
    test_Binarizer()  # 调用 test_Binarizer
