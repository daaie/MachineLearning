
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

# 회귀 모델의 평가
# 사이킷 런의 모든 회귀모델 클래스들은 RegressorMixin 클래스를 상속받음
# RegressorMixin 클래스는 결정계수 R2 점수를 계산하는 score 메소드를 제공
# 결정계수 R2 -1~1 사이의 값을 가지며 , 1에 갈까울수록 좋은 모델임을 확인할 수 있음.

print("모델의 평가 :", model.score(X,y))

#R2 (결정계수)의 계산 공식
# 1 - (실제 정답과 예측 값 차이의 제곱값 합계)/
# (실제 정답과 정답의 평균 값 차이의 제곱값 합계)

pred = model.predict(X)
# 평균을 예측하면 분모 분자 똑같음
# 평균보다 성능이 안좋으면 -값이 나옴
r2 = 1-np.sum((y-pred)**2)/np.sum((y-np.mean(y)) **2)
print("모델의 평가 : ",r2)

# score 메소ㅗ드.
from sklearn.metrics import r2_score
print("모델의 평가(r2_score함수) : ",r2_score(y,pred))