#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/27 9:53
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : pipeline.py
# @Software: PyCharm Community Edition


'''
数据预处理
（pipeline管道流水线）
'''

from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from  sklearn.pipeline import Pipeline


def test_Pipeline(data):
    '''
    测试 Pipeline 的用法
    :param data:  一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train, X_test, y_train, y_test = data
    steps = [("Linear_SVM", LinearSVC(C=1, penalty='l1', dual=False)),
             ("LogisticRegression", LogisticRegression(C=1))]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    print("Named steps:", pipeline.named_steps) #给出流水线上每一步使用的学习器
    print("Pipeline Score:", pipeline.score(X_test, y_test))


if __name__ == '__main__':
    data = load_digits()  # 生成用于分类问题的数据集
    test_Pipeline(model_selection.train_test_split(data.data, data.target, test_size=0.25
                                                    , random_state=0, stratify=data.target))  # 调用 test_Pipeline
