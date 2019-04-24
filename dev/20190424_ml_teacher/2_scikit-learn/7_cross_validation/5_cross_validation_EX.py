# -*- coding: utf-8 -*-

# winequality-red.csv 데이터의 교차 검증 점수를 확인하세요
# 머신러닝 모델은 분류 모델을 사용합니다.

import pandas as pd

fname = '../../data/winequality-red.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:, :-1]
y = wine.iloc[:, -1]

pd.options.display.max_columns = 100
#print(X.info())
#print(X.describe())
#print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, random_state=0, stratify=y.values)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
# 데이터 전처리 시 반드시 학습 데이터에 대해서만
# fit 메소드를 사용합니다.
# (테스트 데이터는 transform 메소드만 사용)
scaler = MinMaxScaler().fit(X_train)

X_train_scaler = scaler.transform(X_train)
X_test_scaler = scaler.transform(X_test)

from sklearn.svm import SVC
# model = SVC(gamma=10)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=1000, random_state=10)

from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
#scores = cross_val_score(model, X_train_scaler, y_train, cv=kfold)
scores = cross_val_score(model, X.values, y.values, cv=kfold)

print("교차 검증 점수(KFold 5) : {}".format(scores))
print("교차 검증 평균 점수(KFold 5) : {:.2f}".format(scores.mean()))


















