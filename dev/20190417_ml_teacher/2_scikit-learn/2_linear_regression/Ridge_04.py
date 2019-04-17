# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:56:00 2019

@author: 502-23
"""

# 사이킷 런의 load_diabestes 함수를 사용하여 당료병 수치를예측할 수 있는 모델을
# 작성한 후 테스트. (LinearRegression, Ridge 클래스를 활용)
# Ridge 클래스의 alpha 값을 조절하여 값의 변화를 확인하세ㅛㅇ.

import pandas as pd
from sklearn.datasets import load_diabetes

diabetes_data = load_diabetes()

X_df = pd.DataFrame(diabetes_data.data)
X_df.columns = diabetes_data.feature_names

X_df.drop('s3', axis=1, inplace = True)
X_df.drop('s1', axis=1, inplace = True)

y_df = pd.DataFrame(diabetes_data.target)

X= X_df.values
y = y_df[0].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, random_state=1)

# 전처리 과정 추가 ###########################################################
#from sklearn.preprocessing import PolynomialFeatures
# 원본 특성 덷이터를 3차 방정식의 형태로 가공하는 
# PolynomialFeatures 객체를 생성.
#poly = PolynomialFeatures(degree = 2, include_bias=False)
#X_train_poly = poly.fit_transform(X_train)
#X_test_poly = poly.transform(X_test)

##############예측기##########################################################

from sklearn.linear_model import LinearRegression, Ridge

#lr_model = LinearRegression().fit(X_train_poly, y_train)
lr_model = LinearRegression().fit(X_train, y_train)
#rg_model = Ridge(0.01).fit(X_train_poly,y_train)
rg_model = Ridge(0.001).fit(X_train,y_train)

#print("LR학습 모델의 스코어 : ", lr_model.score(X_train_poly, y_train))
#print("RG학습 모델의 스코어 : ", rg_model.score(X_train_poly, y_train))

#print("LR테스트 모델의 스코어 : ", lr_model.score(X_test_poly, y_test))
#print("RG테스트 모델의 스코어 : ", rg_model.score(X_test_poly, y_test))

print("LR학습 모델의 스코어 : ", lr_model.score(X_train, y_train))
print("RG학습 모델의 스코어 : ", rg_model.score(X_train, y_train))

print("LR테스트 모델의 스코어 : ", lr_model.score(X_test, y_test))
print("RG테스트 모델의 스코어 : ", rg_model.score(X_test, y_test))


###############그래프를 그려보아요###########################################
from matplotlib import pyplot as plt
coef_range = list(range (1, len(rg_model.coef_) + 1))
plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, rg_model.coef_, 'bo')


plt.hlines(0,1,len(rg_model.coef_) + 1,
           colors = 'y', linestyles = 'dashed')
plt.show()

plt.plot(X[:,0], y, 'bo')
plt.plot(X[:,1], y, 'ro')
plt.plot(X[:,2], y, 'mo')
plt.plot(X[:,3], y, 'go')
plt.plot(X[:,4], y, 'wo')


plt.show()

