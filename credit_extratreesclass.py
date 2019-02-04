from math import *
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
data=pd.read_csv('credit_train.csv')
x=data[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
y=data['default']
clf=ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
clf.fit(x,y)
print(clf.score(x,y))
print(clf.feature_importances_)
test=pd.read_csv('credit_test.csv')
x_test=test[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
y_test=test['default']
print(clf.score(x_test,y_test))
