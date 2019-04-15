# -*- coding: utf-8 -*-

# DecisionTreeClassifier 클래스를 사용하여 load_wind 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.

# DTC 를 이용하여 load_wind 데이터를 분석하고 정확도 및 정밀도 제현율
# DecisionTree의 그래프, 특성 중요도를 시각화하여 확인.

import pandas as pd
from sklearn.datasets import load_wine

load_wine = load_wine()

X_df = pd.DataFrame(load_wine.data)
X_df.columns = load_wine.feature_names
y_df = pd.Series(load_wine.target)

print(y_df.value_counts()/len(y_df))

X= X_df.values
y = y_df.values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=1)

# 결정 트리 알고리즘을 구현하고 있는 DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3, random_state=1).fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(model.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(model.score(X_test, y_test)))
print("특성 중요도:\n{}".format(model.feature_importances_))

import numpy as np
from matplotlib import pyplot as plt

def plot_feature_importances(model):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), load_wine.feature_names)
    plt.xlabel("feature_importances")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)

plot_feature_importances(model)


load_wine.keys()
from sklearn.tree import export_graphviz

export_graphviz(model, out_file='wine_tree.dot', 
                class_names=load_wine.target_names, 
                feature_names=load_wine.feature_names, filled=True)

import graphviz
from IPython.display import display

with open('wine_tree.dot', encoding='utf-8') as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))
