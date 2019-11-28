import pandas as pd

# train = pd.read_csv('data/train.csv', header=0, parse_dates=['date'])
# train_target = pd.read_csv('data/train_label.csv', header=0)
# test = pd.read_csv('data/test.csv', header=0, parse_dates=['date'])
# train_df = pd.merge(train, train_target, on='ID')
#
# test['label'] = -1
# data = pd.concat([train_df, test], axis=0)
# data = data.sort_values(by='ID',ascending=True)
# data.to_csv('df/data.csv', index=False, encoding='utf-8')

data = pd.read_csv('feature1/{}.csv'.format(1), names=['i', 'Feature', 'importance_{}'.format(1), 'fold'])
del data['fold']

for i in range(2, 11):
    r = pd.read_csv('feature1/{}.csv'.format(i), names=['i', 'Feature', 'importance_{}'.format(i), 'fold'])
    del r['i']
    del r['fold']
    print(r)
    data = pd.merge(data, r,on='Feature')

data.to_csv('res/f2.csv',index=False,encoding='utf-8')
