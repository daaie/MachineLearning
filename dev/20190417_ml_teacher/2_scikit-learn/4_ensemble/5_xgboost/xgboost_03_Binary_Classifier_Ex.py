# -*- coding: utf-8 -*-

# diabetes.csv 데이터를 XGBClassifier 를 사용하여
# 분석한 후, 결과를 확인하세요.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
            
fname='../../../data/diabetes.csv'
df = pd.read_csv(fname, header = None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     stratify=y.values, 
                     test_size=0.3, 
                     random_state=21)
    
clf = xgb.XGBClassifier(objective="binary:logistic", 
                         n_estimators=1000,
                         subsample=0.8,
                         reg_lambda=10,
                         #reg_alpha=0,
                        random_state=42)

clf = xgb.XGBClassifier(objective="binary:logistic", 
                         n_estimators=10000,
                         max_depth=5,
                         reg_lambda=1000,
                        random_state=42)
clf.fit(X_train, y_train)


print("학습 결과 : ", clf.score(X_train, y_train))
print("테스트 결과 : ", clf.score(X_test, y_test))


y_pred = clf.predict(X_train)
print("confusion_matrix - 학습데이터")
print(confusion_matrix(y_train, y_pred))

y_pred = clf.predict(X_test)

print("confusion_matrix - 테스트데이터")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
