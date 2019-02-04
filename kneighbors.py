from math import *
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kn
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=kn(n_neighbors=1)
clf.fit(x,y)
print(clf.score(x,y))
t=np.arange(0.0,31.0)
plt.plot(t,y,'--',t,clf.predict(x),'-')
plt.show()
