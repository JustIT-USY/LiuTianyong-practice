import pandas as pd
import numpy as np
import gc, warnings
import pickle
import lightgbm as lgb
import xgboost as xgb
import time
import random, copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)



def show_auc(model, x, y, is_train=True):
    y_pred = model.predict(x)
    score = roc_auc_score(y, y_pred)
    if is_train:
        print('正常值模型 训练集roc_auc_score：{}'.format(score))
    else:
        print('正常值模型 测试集roc_auc_score：{}'.format(score))
    return score

train = pd.read_csv('data/train.csv', parse_dates=['date'])
train_target = pd.read_csv('data/train_label.csv')
test = pd.read_csv('data/test.csv', parse_dates=['date'])
train = pd.merge(train, train_target)

test['label'] = -1
data = pd.concat([train, test])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
print(train.shape, test.shape, data.shape)

col = ['ID', 'date', 'label', 'year', 'month', 'day', 'hour']
features = [i for i in data.columns if i not in col]
print(features)

for fea in features:
    data[fea + '_count'] = data[fea].map(data[fea].value_counts())
    data[fea + '_rank'] = data[fea + '_count'].rank() / float(data.shape[0])
    data[fea] = LabelEncoder().fit_transform(data[fea])

df_train = data[data['label'] != -1].reset_index(drop=True)
df_test = data[data['label'] == -1].reset_index(drop=True)

ignore_feature = ['ID', 'date', 'label', 'E15_rank', 'month', 'year']
feature_name = [i for i in data.columns if i not in ignore_feature]
print(feature_name)
X = df_train
y = df_train['label']
X_test = df_test[feature_name]

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=1, importance_type='gain', metric='auc',
    max_depth=6, learning_rate=0.05, n_estimators=5000,
    objective='binary:logistic', tree_method='exact', subsample=0.7,
    colsample_bytree=0.9, min_child_samples=80,
    n_jobs=-1, reg_lambda=1, min_child_weight=1
)

NFOLDS = 10
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2018)

test_pred_prob = np.zeros((test.shape[0],))
auc = 0
K_auc = []
df_importance_list = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print('Fold:', fold_)

    trn_data, trn_label = X.iloc[trn_idx][feature_name], y.iloc[trn_idx]
    val_data, val_label = X.iloc[val_idx][feature_name], y.iloc[val_idx]
    xgb_model = xgb_model.fit(
        trn_data,
        trn_label,
        eval_set=[(trn_data, trn_label), (val_data, val_label)],
        eval_metric='auc',
        verbose=100,
        early_stopping_rounds=100
    )
    score = show_auc(xgb_model, val_data, val_label, is_train=False)
    K_auc.append(score)
    pre = xgb_model.predict_proba(X_test)[:, 1]
    test_pred_prob += xgb_model.predict_proba(X_test)[:, 1] / folds.n_splits
print(K_auc)
sub = df_test[['ID']]
sub['label'] = test_pred_prob
sub[['ID', 'label']].to_csv('result/sub_xgb.csv', index=False)
