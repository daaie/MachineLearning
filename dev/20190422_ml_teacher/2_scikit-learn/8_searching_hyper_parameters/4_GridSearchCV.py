# -*- coding: utf-8 -*-

# 그리드 서치는 교차검증을 지원하지 안흔ㄴ것 
# 그리드 서치 cv 교차검증 지원하는 것.
# 최적의 하이퍼 파라메터를 검색하기 위한 예제
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, random_state=0)
    
#딕셔너리의 키값 C와 gamma는 파라미터의 이름과 동일 해야함.
param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000],
              'gamma': [0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000]}

#param_grid = [{'kernel': ['rbf'],
#               'C': [0.001, 0.01, 0.1, 1, 10, 100],
#               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
#              {'kernel': ['linear'],
#               'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

print("매개변수 그리드:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(
        SVC(), param_grid, cv=kfold, return_train_score=True, iid=True)
# iid 는 디폴트 false임 kfold 의 특성치가 편중될 경우 false이면 평균을 사용
# iid 가 트루일 경우 편중된 특성에 가중치가 더해짐.
grid_search.fit(X_train, y_train)

# 반환값이 모델ㅇ임 ㅋㅋㅋㅋ그냥 서치하고 모델도 만들고 좋ㅇ구먼 
print("테스트 세트 점수: {:.2f}".format(grid_search.score(X_test, y_test)))

print("최적 매개변수: {}".format(grid_search.best_params_))

print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))

print("최고 성능 모델:\n{}".format(grid_search.best_estimator_))









