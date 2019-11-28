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
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import lightgbm as lgb
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')

def read_data():
    train = pd.read_csv('data/train.csv', header=0, parse_dates=['date'])
    train_target = pd.read_csv('data/train_label.csv', header=0)
    test = pd.read_csv('data/test.csv', header=0, parse_dates=['date'])
    train_df = pd.merge(train, train_target)

    # train_label = train_df[train_df.label == 1]
    # for i in range(1):
    #     train_df = pd.concat([train_df,train_label],axis=0)

    train_df = shuffle(train_df)
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

    '''特征工程结束 <<<<'''

    train = data[data.label != -1]
    test = data[data.label == -1]
    del test['label']

    '''调整特征顺序'''
    l = train['label']
    del train['label']
    train['label'] = l
    return train, test




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
    f_num = len(train.columns) - 1

    del_feature = ['ID', 'label']

    features = [i for i in train.columns if i not in del_feature]

    res_data = pd.DataFrame()
    res_data['ID'] = test['ID']

    train_x = train[features]
    train_y = train['label'].values
    test = test[features]

    params = {
        # 'num_leaves': 255,  # 结果对最终效果影响较大，越大值越好，太大会出现过拟合
        # 'min_data_in_leaf': 30,
        'objective': 'binary:logistic',  # 定义的目标函数
        # 'max_depth': 8,
        'learning_rate': 0.05,
        # "min_sum_hessian_in_leaf": 6,
        'silent': 0,
        "boosting": "gbtree",
        'metric': {'binary_logloss', 'auc'},
        "random_state": 2019,
    }

    K_auc = []
    K = 10
    max_auc = 0

    folds = KFold(n_splits=K, shuffle=True, random_state=98)
    prob_oof = np.zeros((train_x.shape[0],))
    test_pred_prob = np.zeros((test.shape[0],))

    feature_importance_df = pd.DataFrame()
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
        print("fold {}".format(fold_ + 1))

        trn_data = xgb.DMatrix(train_x.iloc[trn_idx], label=train_y[trn_idx])
        val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])

        model = xgb.XGBRegressor(
            learning_rate=0.09,
            n_estimators=800,
            max_depth=3,
            min_child_weight=1,
            gamma=0.4,
            subsample=0.9,
            colsample_bytree=1,
            objective='binary:logistic',
            eval_metric='auc',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
        model.fit(train_x.iloc[trn_idx],train_y[trn_idx])

        prob_oof[val_idx] = model.predict(train_x.iloc[val_idx])

        y_pred = model.predict(train_x.iloc[val_idx])
        auc = roc_auc_score(train_y[val_idx], y_pred)
        K_auc.append(auc)
        print("AUC Score (Test): %f" % auc)

        # fold_importance_df = pd.DataFrame()
        # fold_importance_df["Feature"] = features
        # fold_importance_df["importance"] = clf.feature_importance()
        # fold_importance_df["fold"] = fold_ + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # fold_importance_df.to_csv('feature1/{}.csv'.format(fold_+1))

        test_pred_prob += model.predict(test[features]) / folds.n_splits

        # if auc > max_auc:
        #     max_auc = auc
        #     print("auc>>>>>>", auc)
        #     y_pp = clf.predict(test[features], num_iteration=clf.best_iteration)
        #
        #     res_data['label'] = y_pp
        #     res_data.to_csv('lgb/submission_lgb_10K_{}.csv'.format(auc), index=False, encoding='utf-8')

    print(K_auc)
    print('均值：{}'.format(sum(K_auc) / K))
    # res_data['label'] = test_pred_prob
    # res_data.to_csv('xgb/submission_xgb_10K.csv', index=False, encoding='utf-8')
