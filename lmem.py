import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
x=sm.add_constant(x)
y=data['ice']
est=sm.MixedLM.from_formula("ice ~ temp + street",data,groups=y).fit()
print(est.params)
e=est.params.temp*data['temp']+est.params.street*data['street']+est.params.Intercept
print('R-squared:',r2_score(y,e))
