# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 RandomForestClassifier 클래스를 이용하여 
# 분석한 후, 결과를 확인하세요.

import pandas as pd

fname ='../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

# DataFrame의 모든 데이터를 numpy 배열로 변환
X = X_df.values
# Series 모든 데이터를 numpy 배열로 변환
y = y_df.values
print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=50000,
                                random_state=0).fit(X_train, y_train)

print("model 정확도(학습 데이터) :", model.score(X_train, y_train))
print("model 정확도(테스트 데이터) :", model.score(X_test, y_test))

predicted_1 = model.predict(X_test)

print('Confusion Matrix - 1:')
print(confusion_matrix(y_test, predicted_1))

print('Classification Report - 1 :')
print(classification_report(y_test, predicted_1))

# 각 독립 변수의 중요도(feature importance)를 계산
importances = model.feature_importances_

import numpy as np
from matplotlib import pyplot as plt 

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.title("feature_importances_")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


