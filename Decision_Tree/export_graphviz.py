#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/13 10:10
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : export_graphviz.py
# @Software: PyCharm Community Edition

'''
输出决策图
'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import model_selection
from sklearn.tree.export import export_graphviz


def load_data():
    '''
    加载用于分类问题的数据集。数据集采用 scikit-learn 自带的 iris 数据集
    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    iris = datasets.load_iris()  # scikit-learn 自带的 iris 数据集
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train, test_size=0.25,
                                            random_state=0,
                                            stratify=y_train)  # stratify=y_trai分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4


def export_tree(*data):
    '''
    输出决策图
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # 然后通过Graphviz的dot工具，在命令行中运行命令
    # dot.exe -Tpdf F:/out -o F:/out.pdf生成pdf格式的决策树，
    # 或者执行dot.exe -Tpng F:/out -o F:/out.png来生成陪png格式的决策图。
    # 其中-T选项指定了输出文件格式，-o选项指定了输出文件名
    export_graphviz(clf, "F:/out")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()  # 产生用于分类问题的数据集
    export_tree(X_train, X_test, y_train, y_test)  # 调用 test_DecisionTreeClassifier
