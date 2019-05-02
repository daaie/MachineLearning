# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = LogisticRegression(solver='liblinear', multi_class='ovr')

print("Iris 레이블:\n{}".format(iris.target))

# cross_val_score 함수는 매개변수로 전달된
# 예측기 객체의 타입이 분류인 경우에는 
# 데이터의 서플과정을 선행합니다.
# 반면, 예측기의 타입이 회귀인 경우에는 데이터의 셔플 과정을 생략
scores = cross_val_score(model, iris.data, iris.target, cv=3)
print("교차 검증 점수(cv 3) : {}".format(scores))

# 데이터의 분할을 위해서 사용되는 KFold 클래스
from sklearn.model_selection import KFold

# 생성자의 매개변수 n_splits 값에 지정된 크기만큼 데이터를 
# 분할할 수 있는 기능을 제공
# (기본적으로 데이터를 셔플하지 않고 순차적으로 분할)
# KFold 타입의 객체를 cross_val_score 함수의 cv 매개변수로
# 사용할 수 있음

# KFold 클래스의 객체를 생성할 때, shuffle 매개변수를 지정하지 
# 않는 경우 데이터를 순차적으로 분할하기 때문에 아래와 같이
# 라벨이 정렬된 데이터에는 잘못된 분석 결과가 나올 수 있습니다.
kfold = KFold(n_splits=5)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 5) : {}".format(scores))

kfold = KFold(n_splits=3)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))
























