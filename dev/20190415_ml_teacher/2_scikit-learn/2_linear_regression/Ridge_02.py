# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:08:18 2019

@author: 502-23
"""

# 선형모델의 경우 특성이 많을수록 잘 동작함.^___^

import pandas as pd
from sklearn.datasets import load_boston
############################################################################
boston = load_boston()

# 특성 데이터의 데이터프레임 생성
X_df = pd.DataFrame(boston.data)
X_df.columns = boston.feature_names

# 라벨 데이터의 데이터프레임 생성
y_df = pd.DataFrame(boston.target)

X= X_df.values
y = y_df[0].values
############################################################################

#X, y = load_boston(return_X_y=True)

#print(X.shape)
#print(y.shape)

#############################################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, random_state=1)

print("len(X_train) : ", len(X_train))
print("len(X_test) : ", len(X_test))

from sklearn.linear_model import LinearRegression, Ridge

lr_model = LinearRegression().fit(X_train,y_train)
# 알파의 기본은 1임
ridge_model = Ridge(alpha = 1.7 ).fit(X_train,y_train)

print("LR 학습 모델의 score메소드 :", lr_model.score(X_train,y_train))
print("Ridge 학습 모델의 score메소드 :", ridge_model.score(X_train,y_train))

print("=" * 30)

print("LR 테스트 모델의 score메소드 :", lr_model.score(X_test,y_test))
print("Ridge 테스트 모델의 score메소드 :", ridge_model.score(X_test,y_test))

###############그래프를 그려보아요###########################################
from matplotlib import pyplot as plt
coef_range = list(range (1, len(ridge_model.coef_) + 1))
plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')


plt.hlines(0,1,len(ridge_model.coef_) + 1,
           colors = 'y', linestyles = 'dashed')
plt.show()







