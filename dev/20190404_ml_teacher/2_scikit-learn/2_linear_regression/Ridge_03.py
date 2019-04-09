# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:44:57 2019

@author: 502-23
"""
import pandas as pd

####################1. 데이터 파일 불러오기 #############################
#####특성이 많을수록 LR 의 성능이 올라간다!!!!###########################

fname = '../../data/extended_boston.csv'
df = pd.read_csv(fname, header = None)

####################불러온 파일 X, y로 나누기 ###########################
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, random_state=1)

print("len(X_train) : ", len(X_train))
print("len(X_test) : ", len(X_test))

from sklearn.linear_model import LinearRegression, Ridge

lr_model = LinearRegression().fit(X_train,y_train)
# 알파의 기본은 1임
# 특성이 많으면 많응ㄹ수록 
ridge_model = Ridge(0.088).fit(X_train,y_train)

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


plt.plot(X.values[:,5], y, 'bo')


