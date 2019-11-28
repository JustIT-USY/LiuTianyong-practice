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
import warnings

warnings.filterwarnings('ignore')


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
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour

    data['D1+D2'] = data['D1'] + data['D2']
    data['D1-D2'] = data['D1'] - data['D2']
    # data['D1/D2'] = data['D1'] / data['D2']
    # data['D2/D1'] = data['D2'] / data['D1']
    # data['D1*D2'] = data['D1'] * data['D2']

    # data['D1_square'] = data['D1'] ** 2
    # data['D2_square'] = data['D2'] ** 2

    # data['A_cross'] = data['A1'] * data['A2'] + data['A2'] * data['A3'] + data['A1'] * data['A3']
    # data['B_cross'] = data['B1'] * data['B2'] + data['B2'] * data['B3'] + data['B1'] * data['B3']
    # data['C_cross'] = data['C1'] * data['C2'] + data['C2'] * data['C3'] + data['C1'] * data['C3']

    data['A_square'] = data['A1'] ** 2 + data['A2'] ** 2 + data['A3'] ** 2
    # data['B_square'] = data['B1'] ** 2 + data['B2'] ** 2 + data['B3'] ** 2
    # data['C_square'] = data['C1'] ** 2 + data['C2'] ** 2 + data['C3'] ** 2

    # data['A_sum'] = data['A1'] + data['A2'] + data['A3']
    data['B_sum'] = data['B1'] + data['B2'] + data['B3']
    # data['C_sum'] = data['C1'] + data['C2'] + data['C3']

    data['A_*'] = data['A1'] * data['A2'] * data['A3']
    data['B_*'] = data['B1'] * data['B2'] * data['B3']
    # data['C_*'] = data['C1'] * data['C2'] * data['C3']

    data['A_+'] = data['A1'] + data['A2'] + data['A3']
    data['B_+'] = data['B1'] + data['B2'] + data['B3']
    data['C_+'] = data['C1'] + data['C2'] + data['C3']

    # data['C_ratio'] = data['C1'] * data['C3'] / data['C2']
    # data['A_ratio'] = data['A2'] * data['A3'] / data['A1']

    # data['1_*'] = data['A1'] * data['B1'] * data['C1'] * data['D1']
    # data['2_*'] = data['A2'] * data['B2'] * data['C2'] * data['D2']

    # data['ABCD1_cross'] = data['A1'] * data['B1'] + data['A1'] * data['C1'] + data['A1'] * data['D1']
    # data['ABCD3_cross'] = data['A3'] * data['B3'] + data['A3'] * data['C3']
    #
    # data['E_2-3_ratio'] = data['E2'] / data['E3']
    # data['E_14-1_ratio'] = data['E14'] / data['E1']

    # continuous_columns = ['E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17', 'E19', 'E21', 'E22']
    # data['E_*'] = 1
    # for c in continuous_columns:
    #     data['E_*'] = data['E_*'] * data[c]

    # count_columns = ['E1','E4','E6','E8','E11','E12','E14','E15','E18','E20',
    #                  'E23','E24','E25','E26','E27','E28','E29']
    # data['E_count'] = data[count_columns].sum(axis=1)

    ABC_columns = ['A1_', 'A2_', 'A3_', 'B1_', 'B2_', 'B3_', 'C1_', 'C2_', 'C3_']
    data[ABC_columns] = data[['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]

    normalization_columns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                             'E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17',
                             'E19', 'E21', 'E22',
                             ]

    for column in normalization_columns:
        data[column] = (data[column] - data[column].min(axis=0)) / (data[column].max(axis=0) - data[column].min(axis=0))

    # for i in range(1, 24):
    #     data.loc[data.hour == i, 'hour_click'] = (data[data.hour == i][data.label == 1].shape[0]) / (
    #     data[data.hour == i].shape[0])

    # del_columns = ['B_+', 'E26', 'E16', 'E21', 'E13', 'E25', 'E18', 'E8', 'E15', 'E3', 'D2', 'E22', 'E19']
    #
    # data = data.drop(del_columns, axis=1)
    del data['date']

    col = ['ID', 'date', 'label', 'year', 'month', 'day', 'hour']
    features = [i for i in data.columns if i not in col]
    print(features)

    for fea in features:
        data[fea + '_count'] = data[fea].map(data[fea].value_counts())
        data[fea + '_rank'] = data[fea + '_count'].rank() / float(data.shape[0])
        data[fea] = LabelEncoder().fit_transform(data[fea])

    # E_columns = ['E{}'.format(i) for i in range(1,30)]
    #
    # feature_A = pd.DataFrame()
    # feature_B = pd.DataFrame()
    # feature_C = pd.DataFrame()
    # feature_D = pd.DataFrame()
    # feature_E = pd.DataFrame()
    #
    # feature_A['A_size'] = data.groupby(['A1', 'A2', 'A3']).agg('size')
    # feature_B['B_size'] = data.groupby(['B1', 'B2', 'B3']).agg('size')
    # feature_C['C_size'] = data.groupby(['C1', 'C2', 'C3']).agg('size')
    # feature_D['D_size'] = data.groupby(['D1', 'D2']).agg('size')
    # feature_E['E_size'] = data.groupby(E_columns).agg('size')
    #
    # # data = pd.merge(data, feature_A, on=['A1', 'A2', 'A3'])
    # # data = pd.merge(data, feature_B, on=['B1', 'B2', 'B3'])
    # # data = pd.merge(data, feature_C, on=['C1', 'C2', 'C3'])
    # # data = pd.merge(data, feature_D, on=['D1', 'D2'])
    # # data = pd.merge(data, feature_E, on=E_columns)
    del_columns = ['E18', 'E3_rank', 'E4_rank', 'E5_rank', 'E8_rank', 'E9_count', 'E9_rank', 'E11_rank', 'E12_count',
                   'E12_rank', 'E13_count', 'E13_rank', 'E15_count', 'E15_rank', 'E16_count', 'E16_rank', 'E17_rank',
                   'E18_count', 'E18_rank', 'E19_rank', 'E21_count', 'E21_rank', 'E22_count', 'E22_rank', 'E23_count',
                   'E23_rank', 'E24_count', 'E24_rank', 'E25_count', 'E25_rank', 'E26_count', 'E26_rank', 'E28_count',
                   'E29_count', 'E29_rank', 'B_*_count', 'B_*_rank', 'A_+_count', 'A_+_rank', 'B_+_count', 'B_+_rank',
                   'A1__rank', 'A2__rank', 'A3__rank', 'B1__rank', 'B2__rank', 'B3__rank', 'C2__rank', 'C3__rank']
    data = data.drop(del_columns,axis=1)

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
    model = lgb.LGBMRegressor(
        num_leaves=255,
        max_depth=8,
        learning_rate=0.09,
        # n_estimators=5000,
        objective='binary',
        # min_split_gain=0,
        # min_child_weight=5,
        # min_data_in_leaf=5,
        # max_bin=200,
        # subsample=0.8,
        # subsample_freq=1,
        # colsample_bytree=0.8,
        # seed=1000,
        # n_jobs=-1,
        # # silent=True,
        # lambda_l1=0.1,
        # lambda_l2=0.8
    )

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
    print("特征工程后：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))

    res_data = pd.DataFrame()
    res_data['ID'] = test['ID']

    f_num = len(train.columns) - 1
    x = train.values[:, 1:f_num]
    y = train.values[:, f_num:]
    y = y.flatten()

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=98)
    model = build_model(X_train, y_train)

    score = show_auc(model, X_test, y_test, is_train=False)

    res_set = model.predict(test.values[:, 1:])
    # res_data['label'] = res_set
    # res_data.to_csv('lgb/submission_lgb_63F{}.csv'.format(score), index=False, encoding='utf-8')
