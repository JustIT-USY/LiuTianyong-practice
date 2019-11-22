import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pandas import Series, DataFrame

dataSet = pd.read_table(open('titanic.csv'), sep=',')
print(dataSet)

''' json读取方式（1） '''
f = open('eueo2012.json')
obj = f.read()
result = json.loads(obj)

# 使用dataframe构造器 即可对json完成读取
df1 = DataFrame(result)

''' json读取方式（2) '''
df2 = pd.read_json('eueo2012.json')
