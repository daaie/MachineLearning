# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:42:12 2019

@author: 502-23
"""

from sklearn.datasets import fetch_rcv1

# 멀티라벨 
X,y = fetch_rcv1(return_X_y=True)

import pandas as pd
X_df = pd.DataFrame(X)
y_df = pd.Series(y)

# piperine 클래스와 gridsearchCV 클래스를 활용하여 
# winequlity-red.cs 파일을 분석한 후 결과를 확인하세요.

import pandas as pd
fname = '../../data/winequality-red.csv'
wine = pd.read_csv(fname, sep=';')
X = wine.iloc[:,:-1]
y = wine.iloc[:, -1]


print(type(X_df))
print(type(y_df))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =\
    train_test_split(X.values, y.values, stratify = y.values, random_state=1)


# 2. 데이터의 전처리와 모델의 학습을 자동화 할 수 있는 파이프라인 객체 생성.
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

X_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
#y_pipe = Pipeline([('encoder', LabelEncoder())])
y_encoder = LabelEncoder().fit(y_train)

# 3. 하이퍼 파라메터 검색을 통한 최적의모델 생성.
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {'svm__C':[0.0001,0.001,0.01,0.1,1,10,100,1000],
              'svm__gamma':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
grid = GridSearchCV(X_pipe, param_grid, cv=5, iid= True)
grid.fit(X_train, y_encoder.transform(y_train))


# 4. 학습 된 머신러닝 모델의 평가.
print("학습 결과 : ", grid.score(X_train, y_encoder.transform(y_train)))
print("테스트 결과 :", grid.score(X_test, y_encoder.transform(y_test)))

from sklearn.metrics import classification_report
print("학습의 정밀도, 재현율, f1 점수")
print(classification_report(y_encoder.transform(y_train), grid.predict(X_train)))
from sklearn.metrics import classification_report
print("학습의 정밀도, 재현율, f1 점수")
print(classification_report(y_encoder.transform(y_test), grid.predict(X_test)))