import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf = ExtraTreesRegressor(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(x,y)
p=clf.predict(x)
print(clf.score(x,y))
print(clf.feature_importances_)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],':b')
plt.plot(t,p,'-b')
plt.legend(('real','exrandt'))
plt.show()
