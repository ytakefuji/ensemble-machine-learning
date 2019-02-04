import pandas as pd
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
rbf_feature=RBFSampler(gamma=1,random_state=0,n_components=100)
x_features=rbf_feature.fit_transform(x)
y=data['ice']
f=open('r.txt','wb')
for i in x_features:
 f.write("%s\n" % i)
clf=SGDClassifier()
clf.fit(x_features,y)
print(clf.score(x_features,y))
t=np.arange(0.0,31.0)
plt.plot(t,y,'--',t,clf.predict(x_features),'-')
plt.show()
