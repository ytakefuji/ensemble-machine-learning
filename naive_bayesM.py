from math import *
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=MultinomialNB(alpha=1e-3)
clf.fit(x,y)
print(clf.score(x,y))
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--',t,clf.predict(x),'-')
plt.show()
