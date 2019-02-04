import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf1=SVC(probability=True,kernel='linear')
clf2=AdaBoostClassifier(SVC(C=8000,probability=True,kernel='rbf'),n_estimators=100,learning_rate=1.0,algorithm='SAMME')
clf1.fit(x,y)
clf2.fit(x,y)
p1=clf1.predict(x)
p2=clf2.predict(x)
print(clf1.score(x,y))
print(clf2.score(x,y))
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--b')
plt.plot(t,p1,':b')
plt.plot(t,p2,'-b')
plt.legend(('real','svc','adaBoost'))
#plt.plot(t,data['ice'],':b',t,p1,'-b',t,p2,'--b')
plt.show()
