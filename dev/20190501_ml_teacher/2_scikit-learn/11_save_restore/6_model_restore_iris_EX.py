# -*- coding: utf-8 -*-

from joblib import load
clf = load("../../save/save_model_iris.joblib")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

cancer = load_iris()
# 파이썬이 덕타이핑 기반이기 때문에 필요한 해당 클래스 ()안 불러와도 됨.

X_train, X_test, y_train, y_test = train_test_split(\
     cancer.data, cancer.target, random_state=0)

print("훈련 세트 정확도: {:.3f}".format(clf.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(clf.score(X_test, y_test)))





