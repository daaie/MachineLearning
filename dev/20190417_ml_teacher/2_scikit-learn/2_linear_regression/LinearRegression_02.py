# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 피자 크기에 따른 가격 데이터
# 사이킷런의 모든 예측기는 입력데이터의 형태를
# 2차원 배열로 가정하고 있기 때문에
# 전달할 입력데이터를 reshape를 통해서
# 형태를 변경
X = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
print(X)
# 정답 데이터는 일차원 배열로 선언
y = [7, 9, 13, 17.5, 18]

from sklearn.linear_model import LinearRegression


model = LinearRegression()

model.fit(X,y)

print("평가 점수", model.score(X,y))










plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()
