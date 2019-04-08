# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:47:39 2019

@author: 502-23
"""

import pandas as pd

# 유방암의 음성/양성 데이터
# 이진 분류 데이터 
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_df = pd.DataFrame(cancer.data)
y_df = pd.DataFrame(cancer.target)

print(y_df)

print(X_df.describe())
print(y_df[0].value_counts())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df[0].values, random_state = 1)

print(X_train.shape)
print(X_test.shape)

# 라벨 데이터 확인.
print(y_train.shape)
print(y_test.shape)
print(y_train[:10])
print(y_test[:10])


################최근접 이웃 알고리즘을 구현하고 있는 예측기를 로딩####################
# 예측기 : 사이킷런에서 제공하는 데이터 학습을 위한 클래스.
from sklearn.neighbors import KNeighborsClassifier
# 최근접 알고리즘 예측기는 KNeighborsRegressor 회귀 분류 클래스 .
#  KNeighborsClassifier 분류 클래스 가 있음.

#  n_neighbors 는 데이터와 비교할 데이터 개수.
# 일반적으로 클래스분류에서는 n_neighbors는 홀수여야함. 다수결에 따라 클래스를 나누므로
# 회기분석은 상관없는데 클래스는 다수결이라

# 예측기 객체의 생성
# 사이킷런의 모든 예측기 클래스들은 각 알고리즘에 따라 서로 다른 하이퍼 파라메터를 가지고 있음
# 하이퍼 파라메터 : 사용자가 직접 지정하여 설정하는 값
# 하이퍼 파라메터  에 따라 성능이 변화됨.

model = KNeighborsClassifier(n_neighbors= 5)
# KNeighborsClassifier : k 최근접 이웃 알고리즘을 구현하고 있는 클래스
# 학습이 굉장히 빠른 예측기(단순 저장 , 학습을 안함).
# 예측에 사용될 데이터를 저장하고 있는 데이터와 거리르 ㄹ비교하여 분류를 수행하는 예측기

# 예측기의 데이터 학습
# 사이킷 런의 모든 예측기 클래스들은 fit 메소드를 사용하여 데이터를 학습
# -fit(x,y)의 형태로 사용
# 주의사항
# x(입력데이터)는 반드시 2차원으로 입력
# y(라벨데이터)는 반드시 1차원으로 입력

model.fit(X_train, y_train)

########예측기의 성능평가#################################################################
# 사이킷런의 모든 예측기클래스는 score메소드를 제공
# score 메소드는 예측기의 분류에 따라서 서로 다른 값을 반환
# 분류 모델(Classifier) : 정확도를 반환.
# 회귀 모델(Regressor) : R2 score 반환.
# n_neighbors 가 1일 경우 1=1 매칭이라 학습데이터 ㅓscore가 100일수 밖에없음.

print("학습 데이터 평가: ", model.score(X_train, y_train));
print("학습 데이터 평가: ", model.score(X_test, y_test));


# 최근접 알고리즘은 무조건 정규화를 시켜야함.
# 튀는 값이 있으면 거리를 측저어하기 때문에 score가 좋을 수ㄱ가 없음.


# 예측기를 사용한 예측 결과 반환
# 사이킷 런의 모든 예측기 클래스는 predict 메소드를 제공
# predict 메소드는 2차원 배열의 데이터를 입력받아 예측결과를 1차원으로 반환.

# 앞 3개의 행을 입력
predicted = model.predict(X_test[:10])
print(predicted)
print(y_test[:10])
# train_test_split 할때 각 시드값을 쓰므로 컴퓨터마다 다를 수 이따^__^

# 사이킷 런의 분류를 위한 예측기 클래스는 predict_proba 메소드를 제공
# predict_proba  메소드는 분류할 각 클래스에 대한 입력데이터의 확률값을 반환 (가장 큰 값이 예측값)
predicted_proba = model.predict_proba(X_test[:3])
print(predicted_proba)
#[[0.6 0.4]  [0일 확률, 1일 확률]
# [0.6 0.4]
# [0.  1. ]]






