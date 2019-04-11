# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:45:31 2019

@author: 502-23
"""


import pandas as pd

####################1. 데이터 파일 불러오기 #############################

fname = '../../data/score.csv'
df = pd.read_csv(fname)

df.drop('name', axis=1, inplace = True)

####################불러온 파일 X, y로 나누기 ###########################
X_df = df.iloc[:,1:]
y_df = df.iloc[:,0]


# 특성 데이터의 특징 확인. ##############################################
print(X_df.info())
print(X_df.describe())
print(X_df.shape)
print(y_df.shape)

######학습 ##########################################################
X_train = X_df.values
y_train = y_df.values

# 선형 모델에 L1 제약 조건을 추가한 Lasso 클래스
# L1 제약 조건 : 모든 특성 데이터 중 특정 특성에 대해서만 가중치의 값을 할당하는 제약조건
# 대다수의 특성을 0으로 제약
# L1 제약 조건은 특성 데이터가 많은 데이터를 학습하는 경우
# 빠르게 학습을 할  수 있는 장점을 가짐
# 모든 특성데이터 중 중요도가 높은 특성으르 구ㅂ분.

from sklearn.linear_model import LinearRegression, Ridge, Lasso

lr_model = LinearRegression().fit(X_train,y_train)
ridge_model = Ridge(alpha=5).fit(X_train,y_train)

# 라소도 알파의값이 커질 수록 제약조건이 커짐.
# 알파의 값이 커질 수록 대다 수의 특성들이 0이 됨.
# 알파의 값이 작을 수록 적은 수의 특성들이 0이 됨.
# 릿지는 어지간해서는 0이안됨 라쏘는 조금만 키워도 0ㅇ으로 됨 .
lasso_model = Lasso(alpha=4).fit(X_train,y_train)
print("LR 학습 모델의 score메소드 :", lr_model.score(X_train,y_train))
print("Ridge 학습 모델의 score메소드 :", ridge_model.score(X_train,y_train))
print("Lasso 학습 모델의 score메소드 :", lasso_model.score(X_train,y_train))



###############그래프를 그려보아요###########################################
from matplotlib import pyplot as plt
coef_range = list(range (1, len(ridge_model.coef_) + 1))
plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')
plt.plot(coef_range, lasso_model.coef_, 'go')

plt.hlines(0,1,len(ridge_model.coef_) + 1,
           colors = 'y', linestyles = 'dashed')

plt.show()
# alpha를 키워가면서 확인해보아랏 .
# 선형모델이 너무 학습데이터만 잘맞추면 제약조건을 준다.


























