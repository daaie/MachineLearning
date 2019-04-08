# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:07:06 2019

@author: 502-23
"""

# 사이킷 런의 Load_boston 데이터를 KNN 알고리즘을 사용하여 분석한 후
# 모델의 평가점수를 출력하세요.

import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()

print(boston.keys())

# 특성 데이터의 데이터프레임 생성
X_df = pd.DataFrame(boston.data)
X_df.columns = boston.feature_names

# 라벨 데이터의 데이터프레임 생성
y_df = pd.DataFrame(boston.target)

# 특성 데이터의 샘플 개수 및 결측 데이터 확인
print(X_df.info())

pd.options.display.max_columns = 100

# 특성 데이터의 수치데이터 확인
print(X_df.describe())

# 라벨 데이터 확인
print(y_df.head())
print(y_df.describe())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df[0].values, random_state=1)

print("len(X_train) : ", len(X_train))
print("len(X_test) : ", len(X_test))

# KNeighborsRegressor 클래스
from sklearn.neighbors import KNeighborsRegressor

K = 4
model = KNeighborsRegressor(n_neighbors=K)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f'예측된 타겟 : {predictions}')
print(f'실제  타겟: {y_test}')

print(f'모델 평가(TRAIN) : {model.score(X_train, y_train)}')
print(f'모델 평가(TEST) : {model.score(X_test, y_test)}')



# 회귀모델의 평가에 사용되는 지표##########################################
# r2 score는 -1~1 0이라는건 딱 평균이라는 소리.
# 평균 제곱오차 = (실제값 - 예측한 값)**2 해서 다 더해서 평균 냄.
# 평균 절대오차 = |(실제값 - 예측한 값)| 다 더해서 평균 냄. 


from sklearn.metrics import mean_squared_error, mean_absolute_error


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

#mean_squared_error : 평균 제곱오차의 값

print("학습 데이터에 대한 평균제곱오차 : ", mean_squared_error(y_train,pred_train))
print("테스트 데이터에 대한 평균제곱오차 : ", mean_squared_error(y_test,pred_test))
# 평균제곱오차 텐서플로우 오차함수를 발생시키기 위함..

#mean_absolute_error : 평균 절대오차의 값
print("학습 데이터에 대한 평균절대오차 : ", mean_absolute_error(y_train,pred_train))
print("테스트 데이터에 대한 평균절대오차 : ", mean_absolute_error(y_test,pred_test))



























