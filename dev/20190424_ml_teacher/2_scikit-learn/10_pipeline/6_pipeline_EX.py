# -*- coding: utf-8 -*-

# 1. 데이터의 적재 및 분할
import pandas as pd

fname = '../../data/iris.csv'
iris = pd.read_csv(fname)

X = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, 
                     stratify=y.values, random_state=1)
    
# 2. 데이터의 전처리와 모델의 학습을 자동화할 수 있는
#   파이프라인 객체 생성
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

X_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
y_encoder = LabelEncoder().fit(y_train)

# 3. 하이퍼 파라메터 검색을 통한 최적의 모델 생성
# - 파이프 라인을 사용하여 데이터 전처리 및 
#   머신러닝 모델의 하이퍼 파라메터 검색
from sklearn.model_selection import GridSearchCV

param_grid = {'svm__C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'svm__gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

grid = GridSearchCV(X_pipe, param_grid=param_grid, cv=3, iid=True)
grid.fit(X_train, y_encoder.transform(y_train))

print("Best 교차 검증 점수 : ", grid.best_score_)
print("최적의 하이퍼 파라메터 : ", grid.best_params_)

# 4. 학습된 머신러닝 모델의 평가
print("학습 결과 : ", grid.score(X_train, y_encoder.transform(y_train)))
print("테스트 결과 : ", grid.score(X_test, y_encoder.transform(y_test)))

from sklearn.metrics import classification_report
print("학습 데이터의 정밀도, 재현율, F1")
print(classification_report(y_encoder.transform(y_train), grid.predict(X_train)))
print("테스트 데이터의 정밀도, 재현율, F1")
print(classification_report(y_encoder.transform(y_test), grid.predict(X_test)))




















