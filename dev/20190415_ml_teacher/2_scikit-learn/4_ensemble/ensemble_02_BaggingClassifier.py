# -*- coding: utf-8 -*-

# 보팅 클레스파이어는 여러개의 예측기를 모아다가 하는건데
# 하나의 예측기를 늘려서 쓰는게 배깅.

# 앙상블 방법에서 사용하는 독립적인 예측기의 수가 많을 수록 성능 향상이 
# 일어날 가능성이 높음 
# 다만, 다른 확률 모형을 사용하는데에는 한계가 있기때문에 
# 일반적으로 배깅(bagging) 방법을 사용
# 배깅(bagging) : 같은 예측기를 사용하지만 
# 서로 다른 결과를 출력하는 다수의 예측기를 적용하는 방법
# 동일한 예측기과 데이터를 사용하지만, 부트스트래핑(bootstrapping)과 
# 유사하게 트레이닝 데이터를 랜덤하게 선택해서 다수결 예측기를 적용

# 학습 데이터를 선택하는 방법에 따라
# 같은 데이터 샘플을 중복사용(replacement)하지 않는 경우 Pasting
# 같은 데이터 샘플을 중복사용(replacement)하는 경우 Bagging
# 전체 데이터가 아니라 다차원 독립 변수 중 일부 차원을 
# 선택하는 경우 Random Subspaces
# 전체 데이터 중 일부 샘플과 독립 변수 중 모두 일부만 
# 랜덤하게 사용하는 경우 Random Patches

# BaggingClassifier 클래스를 사용하여 배깅(bagging)을 적용할 수 있음

# base_estimator: 기본 모형
# n_estimators: 모형 갯수. 디폴트 10 -> 늘리면 늘릴수록 학습시간이 늘어남.
# bootstrap: 데이터의 중복 사용 여부. 디폴트 True -> 트레인 데이터가 모형갯수만ㅁ큼 나뉘는데 이때 나뉠때 중복할것인가
# -> false인 경우 서로다른 데이터 셋을이용해서 학습하게 되는것.
# max_samples: 데이터 샘플 중 선택할 샘플의 수 혹은 비율. 디폴트 1.0 - >덜배우개 만들수있는 제약조건
# bootstrap_features: 특징 차원의 중복 사용 여부. 디폴트 False -> 샘플 분할할 때 특성들도 분할함....
# -> 중복이 없으면 서로다른 특성에대해 배움...ㅋ...뭐냐..
# max_features: 다차원 독립 변수 중 선택할 차원의 수 혹은 비율 디폴트 1.0 -> 줄이면 조금 못배우게 만듬.

import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=21)

# 뎁스가 10인 dt 1개를 배움.
#model_1 = DecisionTreeClassifier(max_depth=10, 
#                        random_state=0).fit(X_train, y_train)
model_1 = DecisionTreeClassifier(max_depth=10, 
                        random_state=0).fit(X_train, y_train)

# 뎁스가 3인 dt를 100개 배움, -> 덜배운 예측기가 모였을때 더 일반화.

#model_2 = BaggingClassifier(DecisionTreeClassifier(max_depth=3), 
#                        n_estimators=100, 
#                        random_state=0).fit(X_train, y_train)

model_2 = BaggingClassifier(DecisionTreeClassifier(max_depth=5), 
                        n_estimators=5000, 
                       #bootstrap_features=True,
                        random_state=0).fit(X_train, y_train)



print("model_1 정확도(학습 데이터) :", model_1.score(X_train, y_train))
print("model_2 정확도(학습 데이터) :", model_2.score(X_train, y_train))

print("model_1 정확도(테스트 데이터) :", model_1.score(X_test, y_test))
print("model_2 정확도(테스트 데이터) :", model_2.score(X_test, y_test))

predicted_1 = model_1.predict(X_test)

print('Confusion Matrix - 1:')
print(confusion_matrix(y_test, predicted_1))

print('Classification Report - 1 :')
print(classification_report(y_test, predicted_1))

predicted_2 = model_2.predict(X_test)

print('Confusion Matrix - 1:')
print(confusion_matrix(y_test, predicted_2))

print('Classification Report - 1 :')
print(classification_report(y_test, predicted_2))





# 일반적으로 앙상블은 분류에만.. 회기분석에는 앙상블 잘 사용안함.
# 안그래도 못배우는데 ㅋㅋㅋㅋ 못배우게하는 앙상블로 회기분석하면 더안좋음.









