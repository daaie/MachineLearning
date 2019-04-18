# -*- coding: utf-8 -*-

# winequality-red.csv 데이터의 교차 검증 점수를 확인하세요
# 머신러닝 모델은 분류 모델을 사용합니다.

import pandas as pd

df = pd.read_csv("../../data/winequality-red.csv", sep =';')


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

pd.options.display.max_columns = 100
print(X.info())

print(y.value_counts()/len(y))

# 데이터 분할
# 전체데이터를 이용해서 교차검증을 하면 너무 긍정적인 데이터를 얻을 수 있음
# 학습데이터만으로 하는게 더 일반적임.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X.values, y.values, 
                 stratify=y.values, random_state=21)

# e데이터 전처리
# 반드시 학습 데이터에 대해서만 fit메소드를 부른다
# 테스트 데이터는 transform 만 사용
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler().fit(X_train)

X_train_scalar = scalar.transform(X_train)
X_test_scalar = scalar.transform(X_test)

###########################################################################
#import xgboost as xgb
#model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
#                          n_estimators=10000, max_depth=5, 
#                          reg_lambda=1000, reg_alpha=10)
#

###########################################################################
#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier(max_depth=3, random_state=1)

###########################################################################

#from sklearn.svm import SVC
#model = SVC(gamma='auto')

###########################################################################

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators = 1000, random_state = 0)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train_scalar, y_train, cv=3)
#scores = cross_val_score(model, X, y, cv=3) 
#scores = cross_val_score(model, X_train, y_train, cv=3) 
# 트리구조는 전처리할 필요가 없으므로 그냥 데이터 사용.
print("교차 검증 점수(cv 3) : {}".format(scores))


from sklearn.model_selection import KFold

# kfold 클래스의 객체를 생성할 때 shuffle 매개변수의 값을 true 로 지정하는 경우
# 정답데이터 y data의 비율을 균등하게 포함하고 있는  폴드를 생성할 수 있음.
# 2_cross_validation.png 참고 
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(model, X_train_scalar, y_train, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))

# 예측기 컨텍용도 

# 그래디언트 푸스팅 클래스파이어의 크로스체크.
#[0.60545906 0.60401003 0.58186398] x_train, y_rain
#[0.4953271  0.48217636 0.51789077] x, y
#[0.60297767 0.60401003 0.58186398] x_train_scalar




