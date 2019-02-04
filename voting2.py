import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('red.csv')
x=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y=data['quality']
clf1=GaussianNB()
clf2=ExtraTreesClassifier(n_estimators=82, max_depth=None,min_samples_split=1, random_state=0)
clf3=RandomForestClassifier(random_state=0,n_estimators=250, min_samples_split=1)
clf = VotingClassifier(estimators=[('gnb', clf1), ('et', clf2),('rf',clf3)], voting='soft',weights=[1,8,2]).fit(x,y)
test=pd.read_csv('red_test.csv')
x=test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y=test['quality']
print(clf.score(x,y))
