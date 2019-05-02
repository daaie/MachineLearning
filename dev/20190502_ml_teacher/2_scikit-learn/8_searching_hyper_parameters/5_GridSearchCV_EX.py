# -*- coding: utf-8 -*-

# 최적의 하이퍼 파라메터를 검색하기 위한 예제
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, random_state=0)

# 조건부 매개변수를 사용하기 위한 매개변수 그리드 선언
# 사용 방식 
# - [{조건부 매개변수 1}, {조건부 매개변수 2} ... ]
param_grid = [{'C' : [1, 0.1, 0.01, 0.001, 10, 100, 1000],
              'penalty' : ['l2'],
              'solver' : ['lbfgs']},
                {'C' : [1, 0.1, 0.01, 0.001, 10, 100, 1000],
              'penalty' : ['l1', 'l2'],
              'solver' : ['liblinear']}]

print("매개변수 그리드:\n{}".format(param_grid))

kfold = KFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(
        LogisticRegression(), param_grid, cv=kfold, 
        return_train_score=True, iid=True, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search.score(X_test, y_test)))

print("최적 매개변수: {}".format(grid_search.best_params_))

print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))

print("최고 성능 모델:\n{}".format(grid_search.best_estimator_))









