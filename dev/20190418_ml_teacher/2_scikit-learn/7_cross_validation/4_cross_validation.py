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

# kfold 클래스의 객체를 생성할 때 shuffle 매개변수의 값을 true 로 지정하는 경우
# 정답데이터 y data의 비율을 균등하게 포함하고 있는  폴드를 생성할 수 있음.
# 2_cross_validation.png 참고 
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))

# 예측기 컨텍용도 









