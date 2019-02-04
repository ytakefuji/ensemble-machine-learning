from math import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data=pd.read_csv('ice.csv')
x=data[['temp','street']]
x=sm.add_constant(x)
y=data['ice']
est=sm.OLS(y,x).fit()
print(est.summary())
t=np.arange(0.0,31.0)
e=est.predict(x)
plt.plot(t,y,'-b')
plt.plot(t,e,'--b')
plt.legend(('real','OLS'))
plt.show()
