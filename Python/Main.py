import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# group, labels = kNN.createDataSet()
group, labels = kNN.createDataSet()

print(group)
print(labels)
result = kNN.classify0([0, 0], group, labels, 3)
print(result)

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat,datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# plt.show()
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
plt.show()
kNN.classifyPerson()