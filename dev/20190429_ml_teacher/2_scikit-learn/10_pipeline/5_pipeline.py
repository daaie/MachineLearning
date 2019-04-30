# -*- coding: utf-8 -*-

# 일반적인 머신러닝 단계 
# - 파이프 라인을 사용한 데이터의 전처리 과정 및 
#  머신러닝 모델의 학습 과정 자동화
# - 하이퍼 파라메터를 검색(올바른 방식의 교차 검증을 수행)

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = \
    train_test_split(cancer.data, cancer.target, 
                     stratify=cancer.target, random_state=1)

# 2. 데이터 전처리 및 머신러닝 모델 학습을 위한 
#    파이프 라인 객체를 생성
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', MinMaxScaler()), ("svm", SVC(gamma='scale'))])

# 3. 하이퍼 파라메터 검색을 통한 최적의 모델 생성
# - 파이프 라인을 사용하여 데이터 전처리 및 
#   머신러닝 모델의 하이퍼 파라메터 검색
from sklearn.model_selection import GridSearchCV

# Pipeline을 예측기로 사용하는 GridSearchCV 클래스의
# 파라메터 정보는 키값의 형태를 
# 파이프라인의예측기객체명__파라메터이름
param_grid = {'svm__C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'svm__gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

# GridSearchCV 클래스의 생성자 매개변수로 
# 파이프 라인 객체가 사용될 수 있습니다.
# - 아래의 예는 폴드가 5개로 지정되어 4개의 폴드를 사용하여
#   데이터 정규화를 처리한 후 학습을 진행합니다.
# - 남은 하나의 폴드는 기존의 4개의 폴드로 전처리된 변환기 클래스에
#   의해서 transform 되어 예측에 사용됩니다.
#   (새로운 데이터로 인식되는 방식)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best 교차 검증 점수 : ", grid.best_score_)
print("최적의 하이퍼 파라메터 : ", grid.best_params_)

# 4. 학습된 머신러닝 모델의 평가
print("학습 결과 : ", grid.score(X_train, y_train))
print("테스트 결과 : ", grid.score(X_test, y_test))

from sklearn.metrics import classification_report
print("학습 데이터의 정밀도, 재현율, F1")
print(classification_report(y_train, grid.predict(X_train)))
print("테스트 데이터의 정밀도, 재현율, F1")
print(classification_report(y_test, grid.predict(X_test)))


















