import numpy
from pandas import DataFrame, Series
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset('tips')
print(tips.head())
# 打印数据行列数
print(tips.shape)
# 统计函数
print(tips.describe())
'''
       total_bill         tip        size
count  244.000000  244.000000  244.000000
mean    19.785943    2.998279    2.569672
std      8.902412    1.383638    0.951100
min      3.070000    1.000000    1.000000
25%     13.347500    2.000000    2.000000
50%     17.795000    2.900000    2.000000
'''
print(tips.info())
# 通过info()函数查看是否存在缺失值

# 以消费金额为x轴 小费为y轴 建立散点图 可以发现大致为正相关关系
tips.plot(kind='scatter', x='total_bill',y='tip')
plt.show()

# tips.plot(kind='scatter', x=tips[''],y='tip')
# plt.show()
# 分性别分析进行分析  观察性别是否影响小费金额

# 计算男性小费的平均值
male_tip = tips[tips['sex'] == 'Male']['tip'].mean()
print(male_tip)
# 3.0896178343949043

# 计算女性小费的平均值
famale_tip = tips[tips['sex'] == 'Female']['tip'].mean()
print(famale_tip)
# 2.8334482758620685

# 创建一维数据
s = Series([male_tip, famale_tip], index=['male', 'female'])
print(s)

# 创建一个柱状图
s.plot(kind='bar')
plt.show()
# 由图可得 男性的小费高于女性

''' 小费的时间分部 '''
day = tips['day'].unique()
print(day)

'''[Sun, Sat, Thur, Fri] 只有星期天 星期六 星期四 星期五有小费'''


# 计算星期天小费平均值
sun_tip = tips[tips['day']=='Sun']['tip'].mean()
# 3.2551315789473687

# 计算星期天小费平均值
sat_tip = tips[tips['day']=='Sat']['tip'].mean()
# 2.993103448275862

# 计算星期四小费平均值
thur_tip = tips[tips['day']=='Thur']['tip'].mean()
# 2.771451612903225

# 计算星期五小费平均值
fri_tip = tips[tips['day']=='Fri']['tip'].mean()
# 2.7347368421052627

s = Series([thur_tip, fri_tip, sat_tip, sun_tip], index=['Thur', 'Fri',
                                                         'Sat', 'Sun'])
s.plot(kind='bar')
plt.show()
s.plot(kind='line')
plt.show()

# 建立时间和小费的散点图
tips.plot(kind='scatter',x='day',y='tip')
plt.show()