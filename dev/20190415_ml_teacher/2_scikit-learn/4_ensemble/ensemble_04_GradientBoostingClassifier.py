# -*- coding: utf-8 -*-
# 예측기의 개수는 많은데
# 예측기의 개수가 늘어날 수록 과적합됨.
# 에이다 부스트는 예측기에서 안배운거에 가중치를 줘서 배우는데
# 그래디언ㅇ트 부스팅으느 예측결과의 오차를 줄이는 방향으로 배움 -> 좀더 성능이 좋아보인다.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(\
     cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))