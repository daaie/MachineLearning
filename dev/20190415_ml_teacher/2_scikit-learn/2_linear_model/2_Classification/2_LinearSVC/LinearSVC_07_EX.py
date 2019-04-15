# -*- coding: utf-8 -*-

# LinearSVC 클래스를 사용하여 load_wine 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.

import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
X_df = pd.DataFrame(wine.data)
y_df = pd.Series(wine.target)

print(X_df.describe())
print(y_df.value_counts()/len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values,  stratify=y_df.values, test_size=0.3, random_state=1)

    
# l2패널티를 사용하여 ㄷ데이터를 분석하는 경우
# 특성데이터의 스케일을 압축할 수 있는 데이터 정규화를ㅈ 진행하는 것이
# 성능향상에 도움.    
# L1 패널티를 사용하는 경우 데이터 스케일을 압축할 필요가 없음.
# l1일때는 유의미한 값을 뽑는 것이므로 오히려 정규화를 하는게 더 나쁠 수 있다.
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

from sklearn.svm import LinearSVC
# svc_model = LinearSVC(C=100, max_iter=5000000).fit(X_train, y_train)
# 계속 이터레이터가 부족하다 그러면 그냥 정규화의 문제임.
# 정규화 까먹었다 -> l1 패널티로 바꾸면 유의미한걸 사용함. l2보다 낫다^___^

svc_model = LinearSVC(C=100, max_iter=5000000).fit(X_train, y_train)
#svc_model = LinearSVC(C=100, penalty="l1", dual=False, max_iter=5000000).fit(X_train, y_train)

print("모델 평가(train) : ", svc_model.score(X_train, y_train))
print("모델 평가(test) : ", svc_model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report
pred_train = svc_model.predict(X_train)
pred_test = svc_model.predict(X_test)

print("훈련 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_train, pred_train))
print(classification_report(y_train, pred_train))

print("테스트 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test))









