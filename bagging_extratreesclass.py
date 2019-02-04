import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf1=ExtraTreesClassifier()
clf2= BaggingClassifier(ExtraTreesClassifier(),n_estimators=300,max_samples=0.8, max_features=0.5)
clf1.fit(x,y)
print(clf1.score(x,y))
p=clf1.predict(x)
clf2.fit(x,y)
print(clf2.score(x,y))
q=clf2.predict(x)
t=np.arange(0.0,31.0)
plt.plot(t,y,':b')
plt.plot(t,p,'-b')
plt.plot(t,q,'--b')
plt.legend(('real','extraT','bagging'))
plt.show()
