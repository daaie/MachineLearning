# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:35:36 2019

@author: 502-23
"""

# 일반적인 머신러닝 단계


# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test =\
    train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=1)


# 2. 데이터의 전처리 과정
#   - 라벨 인코딩, 특성데이터의 스케일 조정 드으이 작업을 수행.
#   - 사이킷 런의 변환기 클래스르 활용
#   - fit 메소드는 반드시 학습데이터에만 부름.
#   - transfrom 메소드를 사요핳여 학습 및 테스트 데이터의 변환을 수행.
    # 특히 svc는 스케일링 더 중요함
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler().fit(X_train)
X_train_scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)


# 3. 머신러닝 모델 객체의 생성과 학습
from sklearn.svm import SVC
model = SVC(gamma='scale').fit(X_train_scaled, y_train)


# 4. 학습 된 머신러닝 모델의 평가.
print("학습 결과 : ", model.score(X_train_scaled, y_train))
print("테스트 결과 :", model.score(X_test_scaled, y_test))

from sklearn.metrics import classification_report
print("학습의 정밀도, 재현율, f1 점수")
print(classification_report(y_train, model.predict(X_train_scaled)))
from sklearn.metrics import classification_report
print("학습의 정밀도, 재현율, f1 점수")
print(classification_report(y_test, model.predict(X_test_scaled)))