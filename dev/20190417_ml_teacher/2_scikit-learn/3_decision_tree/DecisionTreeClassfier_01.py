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

# 결정 트리의 노드
# root node: 최상위 노드 
# leap node : 트리르 ㄹ구성하는 가장 하단의 노드( 마지막 노드 )
# pure node : leap node 중 하나의 클래스 값으로 분류된 노드. 
# ( 퓨어노드가 아닌 리프노드는 다수결에 의해 예측 함)

model = DecisionTreeClassifier().fit(X_train, y_train)

print("학습 데이터 정확도 :", model.score(X_train, y_train))
print("테스트 데이터 정확도 :", model.score(X_test, y_test))
