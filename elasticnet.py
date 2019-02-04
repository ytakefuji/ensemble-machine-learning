import pandas as pd
from sklearn.linear_model import ElasticNet
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=ElasticNet(alpha=0.01)
clf.fit(x,y)
print(clf.score(x,y))
