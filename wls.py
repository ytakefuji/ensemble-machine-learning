import pandas as pd
import statsmodels.api as sm
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
x=sm.add_constant(x)
y=data['ice']
est=sm.WLS(y,x).fit()
print(est.summary())
