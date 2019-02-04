import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=KernelRidge(kernel='rbf',alpha=1e-8).fit(x,y)
print(clf.score(x,y))
t=np.arange(0.0,31.0)
plt.plot(t,y,'--',t,clf.predict(x),'-')
plt.show()
