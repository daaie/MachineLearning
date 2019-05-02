# -*- coding: utf-8 -*-

# 최적의 하이퍼 파라메터를 검색하기 위한 예제
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, random_state=0)
    
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

print("매개변수 그리드:\n{}".format(param_grid))

# 교차검증 점수를 기반으로 최적의 파라메터를 검색할 수 있는
# GridSearchCV 클래스
from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=5, shuffle=True)

# GridSearchCV 클래스의 하이퍼 파라메터 정보
# GridSearchCV(예측기 객체, 테스트 파라메터의 딕셔너리 객체, cv=교차검증 폴드 수,...)
# iid 매개변수
# iid 매개변수가 True인 경우 독립 동일 분포라고 가정하여
# 테스트 세트의 샘플 수로 폴드의 점수를 가중 평균함.
# False로 지정하면 단순 폴드 점수의 평균
# - False인 경우 기본 교차 검증과 동작 방식이 동일함
grid_search = GridSearchCV(
        SVC(), param_grid, cv=kfold, return_train_score=True, iid=True)

grid_search.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search.score(X_test, y_test)))

print("최적 매개변수: {}".format(grid_search.best_params_))

print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))

print("최고 성능 모델:\n{}".format(grid_search.best_estimator_))









