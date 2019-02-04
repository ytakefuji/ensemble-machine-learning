import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
rbf_feature=RBFSampler()
x_features=rbf_feature.fit_transform(x)
y=data['ice']
clf=BernoulliNB()
clf.fit(x_features,y)
print(clf.score(x_features,y))
t=np.arange(0.0,31.0)
plt.plot(t,y,'--',t,clf.predict(x_features),'-')
plt.show()
