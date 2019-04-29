# -*- coding: utf-8 -*-

# 일반적인 머신러닝 단계

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = \
    train_test_split(cancer.data, cancer.target, 
                     stratify=cancer.target, random_state=1)
    
# 2. 머신러닝 모델 객체의 생성과 학습
from sklearn.svm import SVC

model = SVC(gamma='scale').fit(X_train, y_train)

# 3. 학습된 머신러닝 모델의 평가
print("학습 결과 : ", model.score(X_train, y_train))
print("테스트 결과 : ", model.score(X_test, y_test))

from sklearn.metrics import classification_report
print("학습 데이터의 정밀도, 재현율, F1")
print(classification_report(y_train, model.predict(X_train)))
print("테스트 데이터의 정밀도, 재현율, F1")
print(classification_report(y_test, model.predict(X_test)))




















