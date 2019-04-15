# -*- coding: utf-8 -*-

import numpy as np

# 피자 크기에 따른 가격 데이터
# 사이킷런의 모든 예측기는 입력데이터의 형태를
# 2차원 배열로 가정하고 있기 때문에
# 전달할 입력데이터를 reshape를 통해서
# 형태를 변경
X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
# 정답 데이터는 일차원 배열로 선언
y = np.array([7, 9, 13, 17.5, 18])


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)

# LinearRegression 클래스의 가중치(기울기), 절편 계산방법
# 1. 입력데이터의 X의 분산 값을 계산
# -  입력데이터X의 산포 수치를 계산
# -  모든 입력데이터가 동일핳다면 분산은 0
# -  분산 수치가 작다면 데이터들이 평균 부근에 밀집
# -  분산 수치가 크다면 데이터들이 평균에서 멀리 위치하고 있음.

# 계산식 : 각 X데이터를 X 데이터의 평균으로 감소시킨 후 제곱하여 합계르 ㄹ구함.
# 합계에 대해서 데이터 개수 -1로 나눈 값이 분산의 값이 됨

X_mean = np.mean(X)
print("x 데이터의 평균 : " , X_mean)

variance = np.sum((X-X_mean) **2)/(len(X)-1)
print("x 데이터의 분산 : " , variance)


# 2. 입력데이터 X와 정답데이터 y사이의 공분산 값을 계산
# - 공분산 : 두개의 변수가 함께 변화하는 수치를 측정하기 위한 방법.
# - 한변수가 증가할 때 다른 변수도 증가한다면 공분산은 양수
# - 한변수가 증가할 때 다른 변수가 감소한다면 공분산은 음수

# 계산식 : 두 변수에 대해서 각각의 ㄷ데이터의 평균만큼 감소시킨 후 서로 
# 곱한 결과의 합계에 대해서 데이터 크기 -1로 나눈 값.
X_mean = np.mean(X) 
y_mean = np.mean(y)

covariance = (X.reshape(-1) - X_mean)*( y - y_mean)
covariance = np.sum(covariance) / (len(y)-1)
print("공분산 수치 : ", covariance)

# 3. X 데이터의 분산 값과 X,y 데이터의 공분산 값을 계산한 후
# 가중치를 계싼함
# -계싼식 : 공분산/분산
weight= covariance/variance 
print("가중치(기울기)",weight)
print("가중치(coef_)",model.coef_)


# 4. 가중치의 값을 계산한 후 절편의 값을 계산
#  정답 데이터(y)의 평균 - (입력데이터(X)의 평균 *가중치)
bias = np.mean(y) - (np.mean(X) * weight)
print("절편",bias)
print("절편(intercept_)",model.intercept_)
