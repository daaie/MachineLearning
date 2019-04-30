# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 GridSearchCV 클래스를 활용하여 분석하세요.
# 활용할 예측기는 SVC, GradientBoostingClassifier 입니다.

# 최적의 하이퍼 파라메터를 검색하기 위한 예제
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

fname = '../../data/winequality-red.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:, :-1]
y = wine.iloc[:, -1]

X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, stratify=y.values, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaler = scaler.transform(X_train)
X_test_scaler = scaler.transform(X_test)

param_grid_svc = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

kfold = KFold(n_splits=5, shuffle=True)

grid_search_svc = GridSearchCV(
        SVC(), param_grid_svc, cv=kfold, 
        return_train_score=True, iid=True, n_jobs=-1)

grid_search_svc.fit(X_train_scaler, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search_svc.score(X_test_scaler, y_test)))
print("최적 매개변수: {}".format(grid_search_svc.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search_svc.best_score_))

print("=" * 33)

param_grid_gbc = {'n_estimators': [100, 1000, 2000],
              'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}

kfold = KFold(n_splits=5, shuffle=True)

grid_search_gbc = GridSearchCV(
        GradientBoostingClassifier(), param_grid_gbc, 
        cv=kfold, return_train_score=True, iid=True, n_jobs=-1)

grid_search_gbc.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search_gbc.score(X_test, y_test)))
print("최적 매개변수: {}".format(grid_search_gbc.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search_gbc.best_score_))













