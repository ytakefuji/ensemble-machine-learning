import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import OrthogonalMatchingPursuit
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
x=sm.add_constant(x)
y=data['ice']
lm=OrthogonalMatchingPursuit(n_nonzero_coefs=3)
est=lm.fit(x,y)
print(est.coef_)
print(est.intercept_)
print(est.score(x,y))
