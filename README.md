# ensemble-machine-learning
This helps readers of a book entitled "Ensemble machine learning" published in Japanese from Kindaikagakusha:
https://www.amazon.co.jp/%E8%B6%85%E5%AE%9F%E8%B7%B5-%E3%82%A2%E3%83%B3%E3%82%B5%E3%83%B3%E3%83%96%E3%83%AB%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92-%E6%AD%A6%E8%97%A4-%E4%BD%B3%E6%81%AD/dp/4764905299/ref=sr_1_1?

Throught the book, the same dataset(ice.csv) is used for introducing conventional statistics methods and ensemble machine learning.
The file ice.csv is composed of four parameters (date, ice, temp, street).
The parameter ice represents sales of the ice cream of the date where temp and street indicate the highest temperature of the day and the number of pedestrians respectively.

The book can be practiced based on Python2.7 on Windows 10. The programs shown here can run on Linux and MacOS. You may be able to use Python3.7 with slight modifications.
The algorithms of the conventional statistics methods are: OLS (ordinary least square: ols.py and olsGUI.py), GLS (generalized least square: gls.py), WLS (weighted least square: wls.py), RLM (robust linear model: rlm.py), GLSAR (feasible generalized least square with autocorrelated: glsar.py), MixedLM (mixed linear model: lmem.py), QuantReg (quantile regression: quantreg.py), OMP (orthogonal matching pursuit: omp.py), ElasticNet (elastic-net machine learning: elasticnet.py). 

Machine learning methods include SVR (support vector regression: svr.py), KRR (kernel ridge regression: krr.py), Naive_Bayes (GaussianNB: naive_bayes.py, MultinomialNB: naive_bayesM.py, BernoulliNB: naive_bayesB.py), DecisionTreeClassifier: decisiontreeclass.py, Kneighbors Regression: knighborsreg.py, KnighborsClassifier: kneighborsclass.py, RadiusNeighborsClassifier: radiusneighbors.py, SGDC (stochastic gradient decendent classifier: sgdc_rbfk.py), Neural network framework (keras: knn.py).

Ensemble methods include Adaboost (classifier: adaboostclass.py, regressor: adaboostreg.py), RandomForest (classifier: randomforestclass.py, regressor: randomforestreg.py), ExtraTrees (classifier: extratreesclass.py, regressor: extratreesreg.py), 
