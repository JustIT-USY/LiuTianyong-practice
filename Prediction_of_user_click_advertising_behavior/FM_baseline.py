import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
K = tf.keras.backend
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from numpy import sort
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


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

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=30, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        a = K.pow(K.dot(x,self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.mean(a-b, 1, keepdims=True)*0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.08),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.09),
                                  )(inputs)
    cross = MyLayer(feature_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.train.AdamOptimizer(0.09),
                  metrics=['binary_accuracy'])
    return model


def build_model(X_train, y_train, X_test, y_test):
    fm = FM(41)

    fm.fit(X_train, y_train, epochs=3, batch_size=128, validation_data=(X_test, y_test))
    return fm


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
