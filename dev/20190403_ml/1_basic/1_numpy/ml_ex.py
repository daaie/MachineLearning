# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:14:40 2019

@author: 502-23
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# 샘플데이터, 타겟데이터
X=np.array([10,20,30,40,50])
Y=np.array([97,202,301,397,505])

# 머신러닝 모델의 객체 생성.
model = LinearRegression()

#머신러닝 모델의 학습 진행
model.fit(X.reshape(-1,1),Y)

pred = model.predict(X.reshape(-1,1))
print("예측결과 : ", pred)
print("예측성능 : ", model.score(X.reshape(-1,1),Y))

from matplotlib import pyplot as plt
plt.scatter(X,Y)
plt.plot(X,pred)
plt.plot(X,Y)
plt.show
