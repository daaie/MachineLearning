# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')


# cross_val_score 교차검증 기능을 제공하는 함수
# 하이퍼 파라메터
# cross_val_score(예측기 객체 , x 데이터, y 데이터 , 교차검증개수)
# 반환되는 값
# - 교차검증 개수에 정의된 크기의 예측기 객체가 생성되며
#  각 예측기의 평가점수가 반환됨
#  회귀모델의 경우 R2 스코어가 반환
#  분류모델은 정활도 반환
#  서로 다른 테스트 데이터와 학습데이터를 가지고 교육을 시켜보는 것 (사진참조)

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)

# 교차검증으로 얻을 수 있는 것은 최저 평균 최대 성능을 알 수 있다.

print("교차 검증 점수 : {}".format(scores))

print("교차 검증 평균 점수 : {:.2f}".format(scores.mean()))