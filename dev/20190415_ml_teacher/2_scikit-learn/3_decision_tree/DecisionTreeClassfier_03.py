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


model = DecisionTreeClassifier(max_depth=3, random_state =1).fit(X_train, y_train)

print("학습 데이터 정확도 :", model.score(X_train, y_train))
print("테스트 데이터 정확도 :", model.score(X_test, y_test))


# 결정트리 모델의 학습 결과를 시각화 하즈아!!
# 설정.
#  1. 현재 운영체제에 맞는 graphviz 설치
#  - http ://www.graphviz.org/
#  2. Path 등록
#  - graphviz 설치된 디렉토리의 bin 경로
#  - 사용자 변수, 시스템 변수 모두에 Path 변수 경로 추가.
#  3. graphviz 파이썬 모듈 설치
#  - pip install graphviz

from sklearn.tree import export_graphviz
export_graphviz(model, 
                out_file='cancer_tree.dot', 
                class_names = ['악성', '양성'],
                feature_names = cancer.feature_names,
                filled=True)


import graphviz 
from IPython.display import display

with open('cancer_tree.dot', encoding = 'utf-8') as f:
        dot_graph = f.read()
        
display(graphviz.Source(dot_graph))