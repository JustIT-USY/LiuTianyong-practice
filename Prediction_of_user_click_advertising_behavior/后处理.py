import pandas as pd

data1 = pd.read_csv('fusedata/73923.csv',encoding='utf-8')
data2 = pd.read_csv('fusedata/73922.csv',encoding='utf-8')

df = pd.merge(data1, data2, on='ID')
df.columns = ['ID','label_1','label_2']

