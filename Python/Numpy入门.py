import numpy as np

data1 = [5, 7, 9, 20]
arr1 = np.array(data1)

data2 = (5, 7, 9, 20)
arr2 = np.array(data2)

data3 = [[1, 2, 3, 4],[5, 6, 7,8]]
arr3 = np.array(data3)

data4 = [1.2, 2, 3.45, 5]
arr4 = np.array(data4)

#创建n维
print(np.zeros((3,4)))
print(np.ones((3,4)))
print(np.empty((2, 2, 2)))

#创建一个像arr1的列表，并且值全为1
np.ones_like(arr1)
#同理
np.zeros_like(arr1)

data = [[1, 2, 3],[2, 4, 5],[3, 5, 7]]
arr = np.array(data)
#秩 即数据轴的个数
arr.ndim
#数据元素个数
arr.size
#数组的维度
arr.shape
#数组中每个元素字节数
arr.itemsize
#创建数组并确定数据类型为float64
arr = np.arange(5,dtype='float64')
#类似强转 但只是返回值 即转化类型赋给一个新的数组  自身不做改变
arr.astype('float32')

arr = np.arange(9)
#改变数组维度
arr.reshape((3,3))

arr1 = np.arange(12).reshape((3,4))
arr2 = np.arange(12, 24).reshape((3,4))

# 散开 / 扁平化
arr2.ravel()
arr2.flatten()

#  数组合并
arr1 = np.arange(12).reshape((3,4))
arr2 = np.arange(12, 24).reshape((3,4))

#竖向拼接
np.concatenate([arr1, arr2], axis=0)
#或者
np.vstack((arr1, arr2))
#横向拼接
np.concatenate([arr1, arr2], axis=1)
#或者
np.hstack((arr1,arr2))

arr5 = np.concatenate([arr1, arr2], axis=0)

#转置
arr1.transpose((1, 0))
arr1.T

arr3 = np.arange(16).reshape((2, 2, 4))
#轴转化
arr3.swapaxes(1, 2)
#生成随机数 100 - 200之间 生成矩阵为 5 * 4
arr6 = np.random.randint(100, 200, size = (5,4))

#生成随机数举证 3 * 5

arr7 = np.random.randn(3, 5)
#以1为均值  2为标准差 3*3的矩阵  正太分部
arr8 = np.random.normal(1, 2,size = (3, 3))

#以2位随机概率发生率 3*3的泊松分部的数组
arr9 = np.random.poisson(2, size = (3,3))

arr10 = np.arange(10).reshape((2,5))

#绝对值
np.abs(arr10)

#平方
np.square(arr10)

''''''
arr12 = np.array([1, 2, 3, 4])
arr13 = np.array([5, 6, 7, 8])
cond = np.array([True,False,False,True])

result = [(x if c else y) for x, y, c in zip(arr12, arr13, cond)]
result = np.where(cond, arr12, arr13)

arr14 = np.random.randn(4,4)
np.where(arr14 > 0,1,0)
# 和
arr14.sum()
# 平均数
arr14.mean()
# 标准差
arr14.std()
# 方差
arr14.var()
# 最小值 最大值
# min max
# 最大索引 最小索引
arr14.argmax()
arr14.argmin()
# 累积和
arr14.cumsum()
# 累积
arr14.cumprod()
#唯一值 即集合
np.unique(arr14)