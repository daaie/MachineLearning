# -*- coding: utf-8 -*-

# Pipeline 클래스와 GridSearchCV 클래스를 활용하여
# winequality-red.csv 파일을 분석한 후 결과를 확인하세요.

import pandas as pd
pd.options.display.max_columns = 100
fname = '../../data/winequality-white.csv'

wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:, :-1]
y = wine.iloc[:, -1]

print(X.info())
print(X.describe())

print(y.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, 
                     stratify=y.values, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', StandardScaler()),
                 ('gbc', GradientBoostingClassifier())])

from sklearn.model_selection import GridSearchCV

param_grid = {'gbc__n_estimators':[100, 500, 1000, 1500],
              'gbc__subsample':[0.5,0.6,0.7,0.8,0.9],
              'gbc__max_depth':[5,6,7,8,9,10],
              'gbc__learning_rate':[0.1, 0.2, 0.3, 0.5]}

grid = GridSearchCV(pipe, param_grid=param_grid, 
                    cv=10, n_jobs=-1).fit(X_train, y_train)
# 폴드가 10개인데 데이터 개수가 5개가 있으면 못맞추게 됨.-> 범주를 줄여야 함
# 와인 퀄리티가 높은것들끼리 , 낮은것들끼리, 중간것들끼리..

print("Best 교차 검증 점수 : ", grid.best_score_)
print("최적의 하이퍼 파라메터 : ", grid.best_params_)

# 4. 학습된 머신러닝 모델의 평가
print("학습 결과 : ", grid.score(X_train, y_train))
print("테스트 결과 : ", grid.score(X_test, y_test))

from sklearn.metrics import classification_report
print("학습 데이터의 정밀도, 재현율, F1")
print(classification_report(y_train, grid.predict(X_train)))
print("테스트 데이터의 정밀도, 재현율, F1")
print(classification_report(y_test, grid.predict(X_test)))



























