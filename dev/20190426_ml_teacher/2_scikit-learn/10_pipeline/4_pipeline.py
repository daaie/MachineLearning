# -*- coding: utf-8 -*-

# 일반적인 머신러닝 단계
# - 파이프 라인을 사용한 데이터의 전처리 과정 및 
#  머신러닝 모델의 학습 과정 자동화

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

# 파이프 라인 객체의 생성
# - Pipeline([(1번째 변환기 클래스 객체의 이름, 객체), (2번째 변환기 클래스 객체의 이름, 객체), ...])
# - Pipeline([1번째 변환기 클래스의 튜플, 2번째 변환기 클래스의 튜플, ...])
# - 파이프 라인의 마지막 객체를 제외한 나머지 객체들은
#   transform, fit_transform 메소드를 제공하는 변환기만 허용
# - 파이프 라인의 마지막 객체는 predict 메소드를 제공하는 
#   예측기 객체가 될 수 있음

# 파이프 라인의 실행 과정
# 1. fit 메소드가 호출되는 경우
# - 입력된 X 데이터를 첫번째 변환기 클래스의 객체로 전달(fit 메소드가 호출)
# - 입력된 X 데이터를 transform 메소드를 통해서 변환 시킴
# - 변환된 입력 데이터 X를 다음에 위치한 변환기 또는 예측기로 전달하여
#   fit 메소드를 실행
# - 다음의 객체가 변환기 객체인 경우 transform 메소드의 실행 결과를 반환하여
#   다음 객체로 전달하고, 예측기 클래스인 경우 실행을 종료

# 2. score/predict 메소드가 호출되는 경우
# - 입력된 X 데이터를 첫번째 변환기 클래스의 객체로 전달(transform 메소드가 호출)
# - 변환된 X 데이터를 다음의 객체로 전달
# - 다음의 객체가 변환기인 경우 다시 한번 transform 메소드가 호출되어 변환된 결과를
#   다음의 객체에게 전달하고
#   만약 예측기 객체인 경우 변환된 X 데이터를 사용하여 score/predict 메소드의
#   실행 결과가 반환
pipe = Pipeline([('scaler', MinMaxScaler()), ("svm", SVC(gamma='scale'))])

# 3. 파이프 라인의 실행을 통한 데이터 전처리 및
#   머신러닝 모델의 학습을 진행
pipe.fit(X_train, y_train)

# 4. 파이프 라인을 통해서 학습된 머신러닝 모델의 평가
print("학습 결과 : ", pipe.score(X_train, y_train))
print("테스트 결과 : ", pipe.score(X_test, y_test))

from sklearn.metrics import classification_report
print("학습 데이터의 정밀도, 재현율, F1")
print(classification_report(y_train, pipe.predict(X_train)))
print("테스트 데이터의 정밀도, 재현율, F1")
print(classification_report(y_test, pipe.predict(X_test)))


















