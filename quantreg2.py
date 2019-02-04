import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
data=sm.add_constant(data)
x=data[['temp','street']]
y=data['ice']
est=smf.quantreg('ice ~ temp + street',data).fit(q=.999)
print(est.summary())
