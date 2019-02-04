import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
rng=np.random.RandomState(1)
clf1=DecisionTreeRegressor(max_depth=4)
clf2=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300,random_state=rng)
clf1.fit(x,y)
clf2.fit(x,y)
p1=clf1.predict(x)
p2=clf2.predict(x)
print clf1.score(x,y)
print clf2.score(x,y)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--b')
plt.plot(t,p1,':b')
plt.plot(t,p2,'-b')
plt.legend(('real','dtree','adaB'))
#plt.plot(t,data['ice'],':b',t,p1,'-b',t,p2,'--b')
plt.show()
