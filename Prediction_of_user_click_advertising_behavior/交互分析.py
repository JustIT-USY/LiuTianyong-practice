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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
from tqdm import tqdm

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
    data['weekday'] = data['date'].dt.weekday

    data['isWeekend'] = 0
    data.loc[data.weekday >= 6, 'isWeekend'] = 1
    # data.loc['isDay'] = 1
    # data.loc[data.hour <= 5,'isDay'] = 0
    # data.loc[data.hour >= 23,'isDay'] = 0

    '''转化率'''
    data['fold'] = data['ID'] % 5
    data.loc[data.label != -1, 'fold'] = 5
    target_feat = []
    cat_list = ['D1', 'D2', 'E4', 'E8', 'E11', 'E12', 'E15', 'E18', 'E25']
    for i in tqdm(cat_list):
        target_feat.extend([i + '_mean_last_1'])
        data[i + '_mean_last_1'] = None
        for fold in range(6):
            data.loc[data['fold'] == fold,i+'_mean_last_1'] = data[data['fold'] == fold][i].map(
                data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['label'].mean()
            )
        data[i+'_mean_last_1'] = data[i+ '_mean_last_1'].astype(float)
    '''
    交叉特征
    '''
    # D A B
    # for i in range(1, 2):
    #     # A
    #     for j in range(1, 4):
    #         # B
    #         for k in range(1, 4):
    #             data['D{}A{}B{}'.format(i, j, k)] = data['D{}'.format(i)] * data['A{}'.format(i)] * data[
    #                 'B{}'.format(i)]

    data['A1_co'] = data['A1'].map(
        lambda x: 0 if x == 6639933182047261336 or x == -7834936860748470404 or x == 4650039790794677128 else 1)
    data['A2_co'] = data['A2'].map(
        lambda x: 0 if x == -266997934034473966 or x == 2774817855788439768 or x == -4298143093126474827 else 1)
    data['A3_co'] = data['A3'].map(
        lambda x: 0 if x == 8246390433496480203 or x == 9040931202870639290 or x == 3424066377286048820 else 1)
    data['B1_co'] = data['B1'].map(
        lambda x: 0 if x == -8639208079192601888 or x == 9054247727413374425 or x == 5535948211432426460 else 1)
    data['B2_co'] = data['B2'].map(
        lambda x: 0 if x == 8626319289109649330 or x == -3043095735331454376 or x == 8388832772844798877 else 1)
    data['B3_co'] = data['B3'].map(
        lambda x: 0 if x == -5533056078568352733 or x == 3840025076341162581 or x == 1794682239533652961 else 1)
    data['C1_co'] = data['C1'].map(
        lambda x: 0 if x == 3524367011807253962 or x == 1961862001648766741 or x == -216616891252271941 else 1)
    data['C2_co'] = data['C2'].map(lambda x: 0 if x == -6705654019294684257 else 1)
    data['C3_co'] = data['C3'].map(
        lambda x: 0 if x == 4285114067042620709 or x == 5818026317047546053 or x == 3285923156243510989 else 1)

    dummy_fea = ['D1', 'D2', 'E3', 'E8', 'E11']
    dummy_df = pd.get_dummies(data.loc[:, dummy_fea], columns=data.loc[:, dummy_fea].columns)
    dunmy_fea_rename_dict = {}
    for per_i in dummy_df.columns.values:
        dunmy_fea_rename_dict[per_i] = per_i + '_onehot'
    print(">>>>>", dunmy_fea_rename_dict)
    dummy_df = dummy_df.rename(columns=dunmy_fea_rename_dict)
    data = pd.concat([data, dummy_df], axis=1)

    data['A1A2'] = data['A1'] / data['A2']
    data['A1A3'] = data['A1'] / data['A3']
    data['A3A2'] = data['A2'] / data['A3']
    data['B1B2'] = data['B1'] / data['B2']
    data['B1B3'] = data['B1'] / data['B3']
    data['E25'] = data['E2'] / data['E7']
    data['E710'] = data['E7'] / data['E10']

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
    # print(features)

    for fea in features:
        data[fea + '_count'] = data[fea].map(data[fea].value_counts())
        data[fea + '_rank'] = data[fea + '_count'].rank() / float(data.shape[0])
        data[fea] = LabelEncoder().fit_transform(data[fea])

    # 离散特征处理
    # print(features)
    one_hot_data = pd.DataFrame()
    one_hot_columns = ['D1', 'D2', 'E4', 'E8', 'E11', 'E12', 'E15', 'E18', 'E25']
    for one_hot_column in one_hot_columns:
        one_hot_data = pd.get_dummies(data[one_hot_column], one_hot_column)
    data = pd.concat([data, one_hot_data], axis=1)

    # E_columns = ['E{}'.format(i) for i in range(1,30)]
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
    # data = pd.merge(data, feature_A, on=['A1', 'A2', 'A3'])
    # data = pd.merge(data, feature_B, on=['B1', 'B2', 'B3'])
    # data = pd.merge(data, feature_C, on=['C1', 'C2', 'C3'])
    # data = pd.merge(data, feature_D, on=['D1', 'D2'])
    # data = pd.merge(data, feature_E, on=E_columns)

    '''特征选择'''
    # 根据自然梯度树特征重要性排序删除 重要性为0的特征
    # del_columns = ['E25_58', 'E25_3', 'D1-D2_count', 'B_+', 'A3_rank', 'E25_rank', 'E25_18', 'E25_21', 'A1A2_count',
    #                'C3_rank', 'C2_co', 'E14_rank', 'E25_66', 'D1+D2', 'E28', 'E25_8', 'E17', 'B1B3_rank', 'E9', 'E11',
    #                'D1_1_onehot', 'C_+_rank', 'A1A3_rank', 'weekday_rank', 'A3A2_rank', 'E27_rank', 'C1__count',
    #                'E1_rank', 'E12', 'E25_6', 'E25_19', 'E6_rank', 'E20_rank', 'A1_rank', 'B1B2_count', 'C3__count',
    #                'E10_count', 'E26', 'B3_rank', 'C2_rank', 'E19', 'E11_1_onehot', 'B1_rank', 'D2', 'A2_co',
    #                'D1_0_onehot', 'E2_count', 'isWeekend_count', 'D1+D2_count', 'E25_4', 'E25_9', 'E25_5', 'A1_co',
    #                'D1_1_onehot_count', 'E3_-3.623288590985344_onehot', 'E8_1_onehot', 'E15', 'E21', 'E3', 'A3_co',
    #                'B3_co', 'E5_count', 'E7_count', 'A_square_count', 'isWeekend', 'D1_count',
    #                'E3_-3.7625716659674406_onehot_count', 'E3_-3.623288590985344_onehot_count', 'E25_41', 'E8', 'E22',
    #                'E4_count', 'E3_-3.8069349285760565_onehot_count', 'D1-D2_rank', 'E11_1_onehot_count', 'A1A2_rank',
    #                'E25_95', 'D1_2_onehot', 'B1B2_rank', 'A3__count', 'E25_98', 'E3_-3.8069349285760565_onehot',
    #                'E11_count', 'E17_count', 'E21_count', 'D1_2_onehot_count', 'D2_1_onehot_count', 'B2_co',
    #                'E3_-3.7625716659674406_onehot', 'C1_co_count', 'D1+D2_rank', 'A_square_rank', 'B1__count', 'E25_23',
    #                'E25_48', 'E25_96', 'E16', 'B1_co', 'D2_0_onehot', 'E10_rank', 'E19_count', 'E19_rank',
    #                'isWeekend_rank', 'D2_1_onehot_rank', 'E3_-3.8069349285760565_onehot_rank',
    #                'E3_-3.787388738614834_onehot_rank', 'E11_2_onehot_count', 'E11_2_onehot_rank', 'B_sum_count',
    #                'A2__count', 'A3__rank', 'B3__rank', 'C2__count', 'E25_51', 'E25_68', 'E25_90', 'E13', 'E18',
    #                'C3_co', 'D1_3_onehot', 'D2_2_onehot', 'E3_-3.818223402328941_onehot',
    #                'E3_-3.8181355542841713_onehot', 'E3_-3.810888090590685_onehot', 'E3_-3.787388738614834_onehot',
    #                'E3_0.27176194202868265_onehot', 'E8_0_onehot', 'E8_2_onehot', 'E8_3_onehot', 'E8_4_onehot',
    #                'E11_0_onehot', 'E11_2_onehot', 'E11_3_onehot', 'D1_rank', 'D2_count', 'D2_rank', 'E2_rank',
    #                'E3_count', 'E3_rank', 'E4_rank', 'E5_rank', 'E7_rank', 'E8_count', 'E8_rank', 'E9_count', 'E9_rank',
    #                'E11_rank', 'E12_count', 'E12_rank', 'E13_count', 'E13_rank', 'E15_count', 'E15_rank', 'E16_count',
    #                'E16_rank', 'E17_rank', 'E18_count', 'E18_rank', 'E21_rank', 'E22_count', 'E22_rank', 'E23_count',
    #                'E23_rank', 'E24_count', 'E24_rank', 'E26_count', 'E26_rank', 'E28_count', 'E28_rank', 'E29_count',
    #                'E29_rank', 'A1_co_count', 'A1_co_rank', 'A2_co_count', 'A2_co_rank', 'A3_co_count', 'A3_co_rank',
    #                'B1_co_count', 'B1_co_rank', 'B2_co_count', 'B2_co_rank', 'B3_co_count', 'B3_co_rank', 'C1_co_rank',
    #                'C2_co_count', 'C2_co_rank', 'C3_co_count', 'C3_co_rank', 'D1_0_onehot_count', 'D1_0_onehot_rank',
    #                'D1_1_onehot_rank', 'D1_2_onehot_rank', 'D1_3_onehot_count', 'D1_3_onehot_rank', 'D2_0_onehot_count',
    #                'D2_0_onehot_rank', 'D2_2_onehot_count', 'D2_2_onehot_rank', 'E3_-3.818223402328941_onehot_count',
    #                'E3_-3.818223402328941_onehot_rank', 'E3_-3.8181355542841713_onehot_count',
    #                'E3_-3.8181355542841713_onehot_rank', 'E3_-3.810888090590685_onehot_count',
    #                'E3_-3.810888090590685_onehot_rank', 'E3_-3.787388738614834_onehot_count',
    #                'E3_-3.7625716659674406_onehot_rank', 'E3_-3.623288590985344_onehot_rank',
    #                'E3_0.27176194202868265_onehot_count', 'E3_0.27176194202868265_onehot_rank', 'E8_0_onehot_count',
    #                'E8_0_onehot_rank', 'E8_1_onehot_count', 'E8_1_onehot_rank', 'E8_2_onehot_count', 'E8_2_onehot_rank',
    #                'E8_3_onehot_count', 'E8_3_onehot_rank', 'E8_4_onehot_count', 'E8_4_onehot_rank',
    #                'E11_0_onehot_count', 'E11_0_onehot_rank', 'E11_1_onehot_rank', 'E11_3_onehot_count',
    #                'E11_3_onehot_rank', 'E710_count', 'E710_rank', 'B_sum_rank', 'A_*_count', 'A_*_rank', 'B_*_count',
    #                'B_*_rank', 'A_+_count', 'A_+_rank', 'B_+_count', 'B_+_rank', 'A1__count', 'A1__rank', 'A2__rank',
    #                'B1__rank', 'B2__count', 'B2__rank', 'B3__count', 'C1__rank', 'C2__rank', 'C3__rank', 'E25_0',
    #                'E25_1', 'E25_2', 'E25_7', 'E25_10', 'E25_11', 'E25_12', 'E25_13', 'E25_14', 'E25_15', 'E25_16',
    #                'E25_17', 'E25_20', 'E25_22', 'E25_24', 'E25_25', 'E25_26', 'E25_27', 'E25_28', 'E25_29', 'E25_30',
    #                'E25_31', 'E25_33', 'E25_34', 'E25_35', 'E25_36', 'E25_37', 'E25_38', 'E25_39', 'E25_40', 'E25_42',
    #                'E25_43', 'E25_44', 'E25_45', 'E25_46', 'E25_47', 'E25_49', 'E25_50', 'E25_52', 'E25_53', 'E25_54',
    #                'E25_55', 'E25_56', 'E25_57', 'E25_59', 'E25_60', 'E25_61', 'E25_62', 'E25_63', 'E25_64', 'E25_65',
    #                'E25_67', 'E25_69', 'E25_70', 'E25_71', 'E25_72', 'E25_73', 'E25_74', 'E25_75', 'E25_76', 'E25_77',
    #                'E25_78', 'E25_79', 'E25_80', 'E25_81', 'E25_82', 'E25_83', 'E25_84', 'E25_85', 'E25_86', 'E25_87',
    #                'E25_88', 'E25_89', 'E25_91', 'E25_92', 'E25_93', 'E25_94', 'E25_97', 'E25_99', 'E25_100']
    #
    # data = data.drop(del_columns, axis=1)

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
    model = GradientBoostingClassifier(n_estimators=10)
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

    columns_list = ['E{}'.format(i) for i in range(1, 30)]

    print("特征工程后：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))

