import pandas as pd

pd_all = pd.read_csv("res.csv", sep=',', header=0)
test_all = pd.read_csv('../data/test.csv',sep=',', header=0)

pd_all = pd.DataFrame(pd_all,columns = ['i','label'])
test_all = pd.DataFrame(test_all,columns=['id','content'])

data = pd.concat([test_all.id,pd_all.label],axis=1)

data[['id','label']].to_csv('result.csv',index=False)



