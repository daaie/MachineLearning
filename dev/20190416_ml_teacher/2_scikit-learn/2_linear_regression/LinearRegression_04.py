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


# 가중치와 절편의 값은 fit메소드 이후에 접근 가능한 멤버.
#print("기울기 : {0}, 절편 : {1}".format(model.coef_, model.intercept_))
model.fit(X,y)

# fit 메소드의 실행 이후에는 학습데이터에 대한 기울기ㅗ아 절편의 값을 확인.
print("기울기 : {0}, 절편 : {1}".format(model.coef_, model.intercept_))

pred_1 = model.predict(X)
#pred_2는 선형 방정식 . predict 함수가 해주는 것.
pred_2 = X.reshape(-1) * model.coef_ + model.intercept_

print(f"preedict 메소드를 사용하여 반환받은 값:{pred_1}")
print(f"가중치(기울기)와 절편을 사용하여 반환받은 값:{pred_2}")















