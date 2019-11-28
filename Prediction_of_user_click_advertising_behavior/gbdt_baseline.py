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
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,GradientBoostingRegressor



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

    data['D1+D2'] = data['D1'] + data['D2']
    data['D1-D2'] = data['D1'] - data['D2']
    data['D1/D2'] = data['D1'] / data['D2']

    data[data['D1/D2'] == np.inf] = -2
    data[data['D1/D2'].isnull()] = 2

    # data['A_sum'] = data['A1'] + data['A2'] + data['A3']
    data['B_sum'] = data['B1'] + data['B2'] + data['B3']
    # data['C_sum'] = data['C1'] + data['C2'] + data['C3']

    data['A_*'] = data['A1'] * data['A2'] * data['A3']
    data['B_*'] = data['B1'] * data['B2'] * data['B3']
    # data['C_*'] = data['C1'] * data['C2'] * data['C3']

    data['A_+'] = data['A1'] + data['A2'] + data['A3']
    data['B_+'] = data['B1'] + data['B2'] + data['B3']
    data['C_+'] = data['C1'] + data['C2'] + data['C3']

    # continuous_columns = ['E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17', 'E19', 'E21', 'E22']
    # data['E_*'] = 1
    # for c in continuous_columns:
    #     data['E_*'] = data['E_*'] * data[c]

    # count_columns = ['E1','E4','E6','E8','E11','E12','E14','E15','E18','E20',
    #                  'E23','E24','E25','E26','E27','E28','E29']
    # data['E_count'] = data[count_columns].sum(axis=1)

    normalization_columns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                             'E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17',
                             'E19', 'E21', 'E22',
                             'B_sum','A_*','B_*','A_+','B_+','C_+'
                             ]
    for column in normalization_columns:
        data[column] = (data[column] - data[column].min(axis=0)) / (data[column].max(axis=0) - data[column].min(axis=0))

    # del_columns = ['D1', 'D2']
    #
    # data = data.drop(del_columns, axis=1)
    del data['date']

    # normalization_columns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
    #                          'E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17',
    #                          'E19', 'E21', 'E22']
    # for column in normalization_columns:
    #     data[column] = (data[column] - data[column].min(axis=0)) / (data[column].max(axis=0) - data[column].min(axis=0))

    # sparse_features = ['D1', 'D2', 'E4', 'E8', 'E11', 'E15', 'E18', 'E25', 'hour']
    # dense_features = ['E1', 'E2', 'E3', 'E5', 'E6', 'E7', 'E9',
    #                   'E10', 'E12', 'E13', 'E14', 'E16', 'E17', 'E16', 'E17',
    #                   'E19', 'E20', 'E21', 'E22', 'E23', 'E24',
    #                   'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'
    #                   ]
    #
    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    #
    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])
    #
    # # 2.count #unique features for each sparse field,and record dense feature field name
    #
    # fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
    #                           for feat in sparse_features] + [DenseFeat(feat, 1, )
    #                                                           for feat in dense_features]

    '''特征工程结束 <<<<'''

    train = data[data.label != -1]
    test = data[data.label == -1]
    del test['label']

    '''调整特征顺序'''
    l = train['label']
    del train['label']
    train['label'] = l
    return train, test


def build_model(x_train, y_train):
    model = GradientBoostingRegressor(n_estimators=10)
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


if __name__ == "__main__":
    train, test = read_data()

    print("特征工程前：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))
    train, test = structural_feature(train, test)
    print(train.isnull())
    print("特征工程后：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))
    f_num = len(train.columns) - 1

    x = train.values[:, 1:f_num]
    y = train.values[:, f_num:]
    y = y.flatten()

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=98)
    model = build_model(X_train, y_train)
    score = show_auc(model, X_test, y_test, is_train=True)

    # res_data = pd.DataFrame()
    # res_data['ID'] = test['ID']
    # res_set = model.predict(test.values[:, 1:])
    # res_data['label'] = res_set
    # res_data.to_csv('lgb/submission_lgb_42F{}.csv'.format(score), index=False, encoding='utf-8')
