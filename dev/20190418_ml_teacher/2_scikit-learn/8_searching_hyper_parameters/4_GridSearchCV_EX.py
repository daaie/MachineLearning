# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:17:31 2019

@author: 502-23
"""
# GridSearchCv클래스를 활용하여 분석하세요
# 활용할 예측기는 SVC 클래스, 그래디언트 클래시파이어 입니다.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC

df = pd.read_csv("../../data/winequality-red.csv", sep =';')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

pd.options.display.max_columns = 100
print(X.info())

print(y.value_counts()/len(y))

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=0)
    
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler().fit(X_train)

X_train_scalar = scalar.transform(X_train)
X_test_scalar = scalar.transform(X_test)

param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000],
              'gamma': [0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000]}

from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=5, shuffle=True)
svc_grid_search = GridSearchCV(
        SVC(), param_grid, cv=kfold, return_train_score=True, iid=True, n_jobs = -1)


from sklearn.ensemble import GradientBoostingClassifier

gc_param_grid = {'n_estimators': [100,1000,2000],
                 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              'subsample': [0.5,0.6, 0.7, 0.8, 0.9]}

gc_grid_search = GridSearchCV(
        GradientBoostingClassifier(), gc_param_grid, cv=kfold, return_train_score=True, iid=True, n_jobs = -1)

svc_grid_search.fit(X_train_scalar, y_train)
gc_grid_search.fit(X_train_scalar, y_train)

print("SVC 테스트 세트 점수: {:.2f}".format(svc_grid_search.score(X_test_scalar, y_test)))
print("SVC 최적 매개변수: {}".format(svc_grid_search.best_params_))
print("SVC 최고 교차 검증 점수: {:.2f}".format(svc_grid_search.best_score_))
print("SVC 최고 성능 모델:\n{}".format(svc_grid_search.best_estimator_))


print("gc 테스트 세트 점수: {:.2f}".format(gc_grid_search.score(X_test_scalar, y_test)))
print("gc 최적 매개변수: {}".format(gc_grid_search.best_params_))
print("gc 최고 교차 검증 점수: {:.2f}".format(gc_grid_search.best_score_))
print("gc 최고 성능 모델:\n{}".format(gc_grid_search.best_estimator_))