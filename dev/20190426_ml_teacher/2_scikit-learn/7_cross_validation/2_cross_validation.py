# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

# 교차검증 기능을 제공하는 cross_val_score 함수
# 하이퍼 파라메터
# cross_val_score(예측기 객체, X 데이터, y 데이터, 교차검증개수)
# 반환되는 값
# - 교차검증 개수에 정의된 크기의 예측기 객체가 생성되며
#   각 예측기의 평가 점수가 반환됨
#   (회귀 모델의 경우 R2 스코어가 반환
#   분류 모델의 경우 정확도가 반환됨)
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)

print("교차 검증 점수 : {}".format(scores))

print("교차 검증 평균 점수 : {:.2f}".format(scores.mean()))