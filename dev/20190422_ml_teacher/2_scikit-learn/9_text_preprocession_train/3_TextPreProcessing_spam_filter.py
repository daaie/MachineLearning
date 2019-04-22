# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:19:29 2019

@author: 502-23
"""

import pandas as pd
fname='../../data/sms.csv'
sms =pd.read_csv(fname)

# 특성데이터가 하나로 이루어져 있기 때문에
# Series타입으로 처리 
X= sms.message
print(type(X))

y= sms.label
print(type(y))
print(y.value_counts()/len(y))
# 데이터 편향이 심함 스팸맞추기 어렵

# 특성(X) 데이터의 전처리 (특성데이터가 중구난방임 긴것도 있고 짧은것도있고 )
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer().fit(X.values)
print("토큰 개수", len(Vectorizer.vocabulary_))
print("변환 결과", Vectorizer.transform([X.values[1]]))
print("변환 결과(밀집행렬)", Vectorizer.transform([X.values[1]]).toarray())

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X.values, y.values, stratify=y.values, random_state=0)

X_train = Vectorizer.transform(X_train_raw)
X_test = Vectorizer.transform(X_test_raw)
# 4180*8711 특성데이터가 엄청 많아짐.

# 선형모델은 특성이 많으면 정확도가 올라감 / 트리구조는 약점임 
# 그래서 로지스틱 레그리션을 선택함 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')


from sklearn.model_selection import cross_val_score, KFold
# 분류작업에서는 셔플을 안해도된다그러나 걍 연습용으로 KFlod함

kfold = KFold(n_splits=10, shuffle = True, random_state = 1)
scores = cross_val_score(model, X_train, y_train, cv=kfold)

print("교차거머증 점수 \n", scores)
print("교차거머증 점수 \n", scores.mean())

model = LogisticRegression(solver='lbfgs',random_state =1).fit(X_train, y_train)
print("학습 결과" , model.score(X_train, y_train))
print("테스트 결과" , model.score(X_test, y_test))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_predict = model.predict(X_train)
print(confusion_matrix(y_train, y_predict))
score = precision_score(y_train, y_predict)
print(score) 
score = recall_score(y_train, y_predict)
print(score) 



from sklearn.metrics import classification_report
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))








