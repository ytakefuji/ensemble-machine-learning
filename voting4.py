import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.ensemble import GradientBoostingClassifier

data=pd.read_csv('red.csv')
x=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y=data['quality']
clf1=ExtraTreesClassifier(n_estimators=82, max_depth=None,min_samples_split=1, random_state=0)
clf2=RandomForestClassifier(random_state=0,n_estimators=250, min_samples_split=1)
clf3=GradientBoostingClassifier(n_estimators=82, learning_rate=0.1,max_depth=1, random_state=0)
clf4=GaussianNB()
clf5=kn(n_neighbors=13)
test=pd.read_csv('red_test.csv')
x_test=test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y_test=test['quality']
clf = VotingClassifier(estimators=[('et', clf1), ('rf', clf2),('gb',clf3),('gnb',clf4),('kn',clf5)], voting='soft',weights=[14, 3, 1, 1, 3]).fit(x,y)
print(clf.score(x_test,y_test))
