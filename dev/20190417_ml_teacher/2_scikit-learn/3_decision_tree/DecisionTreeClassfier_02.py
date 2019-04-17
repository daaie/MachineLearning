# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:56:05 2019

@author: 502-23
"""

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(cancer.data, 
                     cancer.target,  
                     stratify=cancer.target, 
                     test_size=0.3, 
                     random_state=1)
    

# 결정 트리 알고리즘을 구현한 DecisionTreeClassifier
# 트리 구조를 사용하여 데이터를 분류 예측할 수 있는 클래스
    
from sklearn.tree import DecisionTreeClassifier

# 결정 트리 알고리즘을 사용하는 경우 주의사항
# 결정트리 알고리즘은 학습데이터에 과적합 되는 경향을 보입니다.
#- 학습데이터  언젠가는 다맞춘다 ㅋㅋ
# 결정트리 알고리즘을 사용하는 경우 반드시 과적합(Over fitting)을 방지하기 위한 하이퍼 파라메터를 적용
# 과적합(Over Fitting) 을 방지하는 방법
# 1. tree 의 생성을 사ㅈ전에 차단하는 방법
# - 특성 조건(뎁스의개수 지정)을 통해 ㅓ완벽하게 학습하짖 못하도록 방지
# 2. tree의 생성이 완료된 후 특정 leaf 노드들을 삭제하거나 병합하는 방법.
# - 일반적으로 2번이 좋아보이긴한데 사이킷런은 1번의 방법만을 지원.

# default max_depth = none .
# 머신러닝에서 sample은 각각의 행.
# min_sample_leap 는 리프 내부의 개수가 하나인것 ㅋㅋㅋㅋ 데ㅔ이터 하나를 리프하나로 봄.
# random_state = 1 이면 시드가 고정됨
model = DecisionTreeClassifier(max_depth=3, random_state =1).fit(X_train, y_train)

print("학습 데이터 정확도 :", model.score(X_train, y_train))
print("테스트 데이터 정확도 :", model.score(X_test, y_test))
