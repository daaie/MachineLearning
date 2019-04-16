# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 XGBClassifier 클래스를 활용하여
# 분석한 후, 분석 결과를 확인하세요.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
            
fname='../../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     stratify=y.values, 
                     test_size=0.3, 
                     random_state=21)
    
    
model = xgb.XGBClassifier(objective="multi:softprob", 
                          random_state=42,
                          max_depth=10,
                          subsample = 0.7,
                          #reg_lambda=100,
                          #reg_alpha = 0,
                          n_estimators=1000)

# n_estimators 를 높인다 -> 과적합
# 과적합됬는데 못배우니까 subsample 낮춤,,,좀 덜배우게.

model.fit(X_train, y_train)

print("학습 결과 : ", model.score(X_train, y_train))
print("테스트 결과 : ", model.score(X_test, y_test))

y_pred = model.predict(X_train)
print("confusion_matrix - 학습데이터")
print(confusion_matrix(y_train, y_pred))

y_pred = model.predict(X_test)
print("confusion_matrix - 테스트데이터")
print(confusion_matrix(y_test, y_pred))

# 앙상블에서는 과적합 -> 차차 테스트를 맞춰가야함..