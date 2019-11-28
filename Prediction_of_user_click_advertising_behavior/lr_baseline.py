import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from numpy import sort
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression


def read_data():
    train = pd.read_csv('data/train.csv', header=0, parse_dates=['date'])
    train_target = pd.read_csv('data/train_label.csv', header=0)
    test = pd.read_csv('data/test.csv', header=0, parse_dates=['date'])
    train_df = pd.merge(train, train_target)

    return train_df, test


def structural_feature(train, test):
    test['label'] = -1
    data = pd.concat([train, test], axis=0)

    '''特征工程 >>>>>'''
    # data['year'] = data['date'].dt.year
    # data['month'] = data['date'].dt.month
    # data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour

    '''特征工程结束 <<<<'''

    del data['date']

    train = data[data.label != -1]
    test = data[data.label == -1]
    del test['label']

    '''调整特征顺序'''
    l = train['label']
    del train['label']
    train['label'] = l
    return train, test


def build_model(x_train, y_train, X_test, y_test):
    model = LogisticRegression(random_state=20,tol=1e-6)
    model.fit(x_train, y_train)
    return model


def show_auc(model, x, y, is_train=True):
    y_pred = model.predict(x)
    score = roc_auc_score(y, y_pred)
    if is_train:
        print('正常值模型 训练集roc_auc_score：{}'.format(score))
    else:
        print('正常值模型 测试集roc_auc_score：{}'.format(score))
    return score


if __name__ == '__main__':
    train, test = read_data()

    print("特征工程前：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))
    train, test = structural_feature(train, test)
    print("特征工程后：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))

    f_num = len(train.columns) - 1

    x = train.values[:, 1:f_num]
    y = train.values[:, f_num:]
    y = y.flatten()

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=98)
    model = build_model(X_train, y_train, X_test, y_test)

    show_auc(model, X_test, y_test, False)
