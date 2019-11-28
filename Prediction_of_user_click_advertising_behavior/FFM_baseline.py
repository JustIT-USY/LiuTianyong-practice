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

    normalization_columns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                             'E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17',
                             'E19', 'E21', 'E22']
    for column in normalization_columns:
        data[column] = (data[column] - data[column].min(axis=0)) / (data[column].max(axis=0) - data[column].min(axis=0))

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
    def __init__(self, field_dict, field_dim, input_dim, output_dim=30, **kwargs):
        self.field_dict = field_dict
        self.field_dim = field_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.field_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        self.field_cross = K.variable(0, dtype='float32')
        for i in range(self.input_dim):
            for j in range(i+1, self.input_dim):
                weight = tf.reduce_sum(tf.multiply(self.kernel[i, self.field_dict[j]], self.kernel[j, self.field_dict[i]]))
                value = tf.multiply(weight, tf.multiply(x[:,i], x[:,j]))
                self.field_cross = tf.add(self.field_cross, value)
        return self.field_cross

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

def FFM(feature_dim, field_dict, field_dim, output_dim=1):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(1)(inputs)
    cross = MyLayer(field_dict, field_dim, feature_dim, output_dim)(inputs)
    cross = tf.keras.layers.Reshape((1,))(cross)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.train.AdamOptimizer(0.09),
                  metrics=['binary_accuracy','accuracy'])
    return model


def build_model(X_train, y_train, X_test, y_test):
    field_dict = {i: i // 5 for i in range(45)}

    ffm_model = FFM(41, field_dict, 9, 30)
    ffm_model.fit(X_train, y_train, epochs=3, batch_size=128, validation_data=(X_test, y_test))
    print(ffm_model.predict(X_test))

    return ffm_model


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
