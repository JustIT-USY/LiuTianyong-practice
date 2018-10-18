from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

obj = Series([1, -2, 3, -4])

obj2 = Series([1, -2, 3, -4], index=['a', 'b', 'c', 'd'])
obj2.values
obj2[['a', 'b']]
np.abs(obj2)

data = {
    '张三':92,
    '李四':78,
    '王五':68,
    '小明':82
}
obj3 = Series(data)

names = ['张三', '李四', '王五', '小明']
obj4 = Series(data, index=names)
obj4.name = 'math'
obj4.index.name = 'student'

data = {
    'name':['张三', '李四', '王五', '小明'],
    'sex':['female', 'female', 'male', 'male'],
    'year':[2001, 2001, 2003, 2202],
    'city':['北京', '上海', '广州', '北京']
}
df = DataFrame(data)
df = DataFrame(data, columns=['name', 'sex', 'year', 'city'],
               index=['a', 'b', 'c', 'd'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

obj = Series([1, -2, 3, -4], index=[0, 2, 3,5])
obj2 = obj.reindex(range(6), method='ffill')
df = DataFrame(np.arange(9).reshape(3, 3),index=['a', 'c', 'd'],
               columns=['name', 'id', 'se'])
df2 = df.reindex(['a', 'b', 'c', 'd'])
df3 = df.reindex(columns=['name', 'year', 'id'], fill_value=0)
data2 = {
    'name':['张三', '李四', '王五', '小明'],
    'grade':[68, 78, 63, 92]
}
df = DataFrame(data2)
df2 = df.sort_values(by='grade')


new_data = {
    'city':'武汉',
    'name':'小李',
    'sex':'male',
    'year':2002
}
df = df.append(new_data, ignore_index=True)  #或略索引值
'''
  name  grade city   sex    year
0   张三   68.0  NaN   NaN     NaN
1   李四   78.0  NaN   NaN     NaN
2   王五   63.0  NaN   NaN     NaN
3   小明   92.0  NaN   NaN     NaN
4   小李    NaN   武汉  male  2002.0
'''

# 增加列
df['class'] = 2018
'''
  name  grade city   sex    year  class
0   张三   68.0  NaN   NaN     NaN   2018
1   李四   78.0  NaN   NaN     NaN   2018
2   王五   63.0  NaN   NaN     NaN   2018
3   小明   92.0  NaN   NaN     NaN   2018
4   小李    NaN   武汉  male  2002.0   2018
'''
df['math'] = [92, 78, 58, 69, 82]
'''
  name  grade city   sex    year  class  math
0   张三   68.0  NaN   NaN     NaN   2018    92
1   李四   78.0  NaN   NaN     NaN   2018    78
2   王五   63.0  NaN   NaN     NaN   2018    58
3   小明   92.0  NaN   NaN     NaN   2018    69
4   小李    NaN   武汉  male  2002.0   2018    82
'''
# 删除第二行
new_df = df.drop(2)
'''
  name  grade city   sex    year  class  math
0   张三   68.0  NaN   NaN     NaN   2018    92
1   李四   78.0  NaN   NaN     NaN   2018    78
3   小明   92.0  NaN   NaN     NaN   2018    69
4   小李    NaN   武汉  male  2002.0   2018    82
'''
# 修改
new_df.rename(index={3:2,4:3},columns = {'math':'Math'},inplace=True)
'''
  name  grade city   sex    year  class  Math
0   张三   68.0  NaN   NaN     NaN   2018    92
1   李四   78.0  NaN   NaN     NaN   2018    78
2   小明   92.0  NaN   NaN     NaN   2018    69
3   小李    NaN   武汉  male  2002.0   2018    82
'''

obj1 = Series([3.2, 5.3, -4.4, -3.7], index=['a', 'b', 'c', 'd'])
obj2 = Series([6.0, -2.4, 4.4, 3.4], index=['a', 'b', 'c', 'd'])
obj1 + obj2
# a    9.2
# b    2.9
# c    0.0
# d   -0.3
# dtype: float64

df1 = DataFrame(np.arange(9).reshape(3, 3), columns=['a', 'b', 'c'],index=['apple', 'tea', 'banana'])
df2 = DataFrame(np.arange(9).reshape(3, 3),columns=['a', 'b', 'd'],index=['apple', 'tea', 'coco'])
df1 + df2
'''
          a    b   c   d
apple   0.0  2.0 NaN NaN
banana  NaN  NaN NaN NaN
coco    NaN  NaN NaN NaN
tea     6.0  8.0 NaN NaN
'''

data = {
    'fruit':['apple', 'orange', 'grape', 'banana'],
    'price':['25元', '42元', '35元', '4元']
}
df1 = DataFrame(data)
def f(x):
    return x.split('元')[0]
df1['price'] = df1['price'].map(f)

df2 = DataFrame(np.random.randn(3, 3), columns=['a', 'b', 'c'],index=['app', 'win', 'mac'])
f = lambda x:x.max()-x.min()
df2.apply(f)
obj1 = Series([-2, 3, 2, 1],index=['b', 'a', 'd', 'c'])

df = DataFrame(np.arange(0,9).reshape(3, 3),columns=['a', 'b', 'c'])
# 按列计算和
df.sum()
# 按行计算和
df.sum(axis = 1)
# 统计函数
df.describe()
obj = Series(['a', 'b', 'c', 'a', 'b'])

'''分层索引'''
obj = Series(np.random.randn(9),
             index=[['one', 'one', 'one', 'two', 'two', 'two', 'three','three','three'],
                    ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']])
df5 = DataFrame(np.arange(16).reshape(4, 4),index=[['one', 'one', 'two', 'two'],
                                                   ['a', 'b', 'a', 'b']],
                columns=[['apple', 'apple', 'orange', 'orange'],
                         ['red', 'green', 'red', 'green']])
df5['apple']


'''pandas可视化'''
# 服从正态分布
s = Series(np.random.normal(size = 10))
s.plot()