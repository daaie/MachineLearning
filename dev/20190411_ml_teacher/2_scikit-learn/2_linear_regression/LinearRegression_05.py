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

# 입력 데이터에 대한 정답 데이터(수치)를
# 선형모델을 사용하여 예측할 수 있는 
# LinearRegression 모델 클래스를 import
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)


# LinearREgression 클래스의 비용함수
# (머신러닝 모델의 손실(비용)함수는 --학습의 정확도를 판단하는 기준이 됨.)
# 머신러닝 모델의 학습 완성도를 판단하는 기준.
# (성능이 좋은 머신러닝 모델은 손실(비용)함수의 결과가 작음 - 일반적으로)
# 잔차 : 훈련데이터를 통해서 예측된 결과와 실제 정답사이의 오차.
# LinearRegression 클래스의 모델은 잔차의 합계를 최소화 할 수 있는 가중치, 절편을
# 찾아낸ㄴ 것이 최종 목표 

# 오차 계산 방법
# 모델의 예측결과와 ㅈ실제 정답사이의 오차값을 계산한 후 제곱한 값으 합계를 구함.
# 합계의 평균 오차로 사용. == > 평균 제곱오차
# 평균 제곱오ㅗ차가 가장 작은 값을 찾는거 ( 가중치와 절편을 이용해 )
# (모델의 예측결과 - 실제 정답)**2 의 평균값.

# 1.예측결과 반환.
pred = model.predict(X)
# 2. 예측 결과와 정답 사이의 오차를 계산한 후 제곱
loss = (pred - y)**2
# 3. 제곱된 값의 합계를 구한 후 평균값응ㄹ 반환.
loss_avg = np.mean(loss)

print("모델의 오차 값(잔차 제곱의 합계 평균)", loss_avg)


from sklearn.metrics import mean_squared_error
print("모델의 평균제곱오차", mean_squared_error(y,pred))
