# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:48:35 2019

@author: 502-23
"""

# GridSearchCv클래스를 사용하여
# sms.csv 파일을 분석할 수 있는 최적의 모델을 검색하여 분석결과를 출력하세요.

################################################################################

import pandas as pd
fname='../../data/sms.csv'
sms =pd.read_csv(fname)
X= sms.message
y= sms.label

################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer().fit(X.values)
vectorizer = TfidfVectorizer(stop_words='english').fit(X.values)
vectorizer = TfidfVectorizer(min_df=2).fit(X.values)

print("토큰 개수",len(vectorizer.vocabulary_))
print("변환 결과", vectorizer.transform([X.values[1]]))

################################################################################

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X.values, y.values, stratify=y.values, random_state=0)

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

################################################################################

param_grid = [{'kernel': ['rbf','linear'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}]


print("매개변수 그리드:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC

kfold = KFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(
        SVC(), param_grid, cv=kfold, return_train_score=True, iid=True)

grid_search.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search.score(X_test, y_test)))
print("최적 매개변수: {}".format(grid_search.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))
print("최고 성능 모델:\n{}".format(grid_search.best_estimator_))


################################################################################

# 1000 제약조건을 풀어주는 것 (C가 커지면 )
param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

print("매개변수 그리드:\n{}".format(param_grid))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
model = LogisticRegression(solver ='lbfgs',max_iter=10000)
kfold = KFold(n_splits=10, shuffle=True)
grid_model = GridSearchCV(model, param_grid = param_grid, cv=kfold, n_jobs=-1, return_train_score=True).fit(X_train,y_train)


print("테스트 세트 점수: {:.2f}".format(grid_model.score(X_test, y_test)))
print("최적 매개변수: {}".format(grid_model.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_model.best_score_))
print("최고 성능 모델:\n{}".format(grid_model.best_estimator_))




















