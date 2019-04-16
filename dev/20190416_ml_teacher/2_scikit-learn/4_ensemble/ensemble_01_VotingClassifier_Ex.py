# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 LogisticRegression, KNeighborsClassifier, 
# GaussianNB 를 조합한 VotingClassifier로 분석한 후, 결과를 확인하세요.


import pandas as pd

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
model3 = DecisionTreeClassifier( random_state=1).fit(X_train, y_train)
ensemble = VotingClassifier(estimators=[('lr', model1), ('knn', model2), ('dt', model3)],
                                        voting='soft')


model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

ensemble.fit(X_train, y_train)

print("LR훈련 세트 정확도: {:.3f}".format(model1.score(X_train, y_train)))
print("LR테스트 세트 정확도: {:.3f}".format(model1.score(X_test, y_test)))
print("KNN훈련 세트 정확도: {:.3f}".format(model2.score(X_train, y_train)))
print("KNN테스트 세트 정확도: {:.3f}".format(model2.score(X_test, y_test)))
print("DT훈련 세트 정확도: {:.3f}".format(model3.score(X_train, y_train)))
print("DT테스트 세트 정확도: {:.3f}".format(model3.score(X_test, y_test)))
print("EB훈련 세트 정확도: {:.3f}".format(ensemble.score(X_train, y_train)))
print("EB테스트 세트 정확도: {:.3f}".format(ensemble.score(X_test, y_test)))