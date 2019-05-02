# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = LogisticRegression(solver='liblinear', multi_class='ovr')

print("Iris 레이블:\n{}".format(iris.target))

scores = cross_val_score(model, iris.data, iris.target, cv=3)
print("교차 검증 점수(cv 3) : {}".format(scores))

from sklearn.model_selection import KFold

# KFold 클래스의 객체를 생성할 때, shuffle 매개변수의 값을
# True 로 지정하는 경우 정답 데이터(y)의 비율을 균등하게
# 포함하는 폴드들을 생성할 수 있습니다.
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))
















