import pandas as pd

data1 = pd.read_csv('fusedata/20191126-161259315.csv', encoding='utf-8')
data2 = pd.read_csv('fusedata/submission_lgb_10K_0.7306372823176981.csv', encoding='utf-8')
# data3 = pd.read_csv('fusedata/submission_lgb_10K_0.7290705900077424.csv', encoding='utf-8')
# data4 = pd.read_csv('fusedata/submission_lgb_10K_0.7290384791906701.csv', encoding='utf-8')
# data5 = pd.read_csv('fusedata/submission_lgb_10K_0.7290304334360973.csv', encoding='utf-8')

df = pd.merge(data1, data2, on='ID')
# df = pd.merge(df, data3, on='ID')
# df = pd.merge(df, data4, on='ID')
# df = pd.merge(df, data5, on='ID')
df.columns = ['ID', '1', '2']
res = pd.DataFrame()
res['ID'] = df['ID']
res['label'] = df['1'] * 0.8 + df['2'] *0.2
res.to_csv('融合/融合5.csv', index=False, encoding='utf-8')
