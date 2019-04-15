# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 LogisticRegression, KNeighborsClassifier, 
# GaussianNB 를 조합한 VotingClassifier로 분석한 후, 결과를 확인하세요.


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

fname = '../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X_df = df.iloc[:, :-1]
print(type(X_df))

y_df = df.iloc[:, -1]
print(type(y_df))

# DataFrame의 모든 데이터를 numpy 배열로 변환
X = X_df.values
# Series 모든 데이터를 numpy 배열로 변환
y = y_df.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(solver='lbfgs')
model2 = KNeighborsClassifier(n_neighbors=3)
model3 = DecisionTreeClassifier(max_depth=3, random_state=1).fit(X_train, y_train)
ensemble = VotingClassifier(estimators=[('lr', model1), ('knn', model2), ('dt', model3)],
                                        voting='soft')


print("훈련 세트 정확도: {:.3f}".format(ensemble.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(ensemble.score(X_test, y_test)))