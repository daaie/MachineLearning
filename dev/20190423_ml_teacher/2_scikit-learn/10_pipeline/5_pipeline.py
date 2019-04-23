# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:35:36 2019

@author: 502-23
"""

# 일반적인 머신러닝 단계
# - 파이프라인을 활용한 데이터 전처리 및 하이퍼파라메터 검색.
# - 원래 3,4번에서 쓰던 방식은 정규화후에 그리스써치를 한다 -> 그리드 서치 내에서는 validation을 하는데
# - 이 validation은 정규화를 끝낸 데이터 내에서 하므로 정답에 영향을 미친다.


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test =\
    train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=1)


from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', MinMaxScaler()),('svm',SVC(gamma='scale'))])
pipe.fit(X_train, y_train)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Pipeline 을 예측기로 사용하는 GridSearchCV 클래스의
# 파라메터 정보는 키값으 형태를 파이프라인의 예측기 객체__파라메터이름으로 할 수 있다. 

param_grid = {'svm__C':[0.0001,0.001,0.01,0.1,1,10,100,1000],
              'svm__gamma':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

# fit을 호출하는 순간 x_train 전체 데이터에 대해 cv(폴드)에 의해 나뉘어진 데이터가 pipe에 전달됨.
# x_train을 5개로 나누고 4개를 피팅 1개를 검증
# 4개를 학습으로 전달 / 한개를 검증데이터로 전달.
# GridSearchCV 클래스의 생성자 매개변수로 
# 파이프 라인 객체가 사용될 수 있습니다.
# - 아래의 예는 폴드가 5개로 지정되어 4개의 폴드를 사용하여
#   데이터 정규화를 처리한 후 학습을 진행합니다.
# - 남은 하나의 폴드는 기존의 4개의 폴드로 전처리된 변환기 클래스에
#   의해서 transform 되어 예측에 사용됩니다.
#   (새로운 데이터로 인식되는 방식)
grid.fit(X_train, y_train)


print("베스트 교차 검증 점수: ", grid.best_score_)
print("베스트 교차 검증 파라메터", grid.best_params_)
