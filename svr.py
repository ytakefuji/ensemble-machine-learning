import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import re,os
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=SVR(kernel='rbf',C=1e7,epsilon=.01,max_iter=-1,tol=1e-7,verbose=1,gamma=10.1).fit(x,y)
print(clf)
'''
print(clf.score(x,y))
t=np.arange(0.0,31.0)
plt.plot(t,y,'--',t,clf.predict(x),'-')
plt.show()
'''
