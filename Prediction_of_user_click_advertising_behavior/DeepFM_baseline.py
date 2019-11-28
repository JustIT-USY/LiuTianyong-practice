import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names


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
    del data['date']

    data['D1+D2'] = data['D1'] + data['D2']
    data['D1-D2'] = data['D1'] - data['D2']
    data['D1/D2'] = data['D1'] / data['D2']

    # data['A_sum'] = data['A1'] + data['A2'] + data['A3']
    data['B_sum'] = data['B1'] + data['B2'] + data['B3']
    # data['C_sum'] = data['C1'] + data['C2'] + data['C3']

    data['A_*'] = data['A1'] * data['A2'] * data['A3']
    data['B_*'] = data['B1'] * data['B2'] * data['B3']
    # data['C_*'] = data['C1'] * data['C2'] * data['C3']

    data['A_+'] = data['A1'] + data['A2'] + data['A3']
    data['B_+'] = data['B1'] + data['B2'] + data['B3']
    data['C_+'] = data['C1'] + data['C2'] + data['C3']

    normalization_columns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                             'E2', 'E3', 'E5', 'E7', 'E9', 'E10', 'E13', 'E16', 'E17',
                             'E19', 'E21', 'E22']
    for column in normalization_columns:
        data[column] = (data[column] - data[column].min(axis=0)) / (data[column].max(axis=0) - data[column].min(axis=0))

    sparse_features = ['D1', 'D2', 'E4', 'E8', 'E11', 'E15', 'E18', 'E25', 'hour']
    dense_features = ['E1', 'E2', 'E3', 'E5', 'E6', 'E7', 'E9',
                      'E10', 'E12', 'E13', 'E14', 'E16', 'E17', 'E16', 'E17',
                      'E19', 'E20', 'E21', 'E22', 'E23', 'E24',
                      'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'
                      ]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    '''特征工程结束 <<<<'''

    train = data[data.label != -1]
    test = data[data.label == -1]
    del test['label']

    '''调整特征顺序'''
    l = train['label']
    del train['label']
    train['label'] = l
    return train, test, feature_names, linear_feature_columns, dnn_feature_columns


if __name__ == "__main__":
    train, test = read_data()

    '''离散 D1	D2	E4	E8	E11	E15 E18	E25'''
    print("特征工程前：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))
    train, test, feature_names, linear_feature_columns, dnn_feature_columns = structural_feature(train, test)
    print("特征工程后：训练集：shape:{0}  测试集：shape：{1}".format(train.shape, test.shape))

    # 3.generate input data for model

    train, test = train_test_split(train, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train['label'].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("test LogLoss", round(log_loss(test['label'].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test['label'].values, pred_ans), 4))
