# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:28:00 2019

@author: 502-23
"""

# data 디렉토리에 있는 sms.csv 파일을 분서갛여 결과를 확인
# 말뭉치 변환에 TfidfVectorizer 클래스를 활용하세요.
# CountVectorizer 를 0~1사이에 압축해서 값을 표현함.

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
fname='../../data/sms.csv'
sms =pd.read_csv(fname)
X= sms.message
y= sms.label

################################################################################

vectorizer = TfidfVectorizer().fit(X.values)
vectorizer = TfidfVectorizer(stop_words='english').fit(X.values)
vectorizer = TfidfVectorizer(min_df=3).fit(X.values)

print("토큰 개수",len(vectorizer.vocabulary_))
#print("토큰 내용",(vectorizer.vocabulary_))
print("변환 결과", vectorizer.transform([X.values[1]]))

################################################################################

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X.values, y.values, stratify=y.values, random_state=0)

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

################################################################################

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train)

print("교차검증 점수(정확도) \n", scores)
print("교차검증 점수(정확도 - 평균) \n", scores.mean())

################################################################################

scores = cross_val_score(model, X_train, y_train, scoring='precision')

print("교차검증 점수(정밀도) \n", scores)
print("교차검증 점수(정밀도 - 평균) \n", scores.mean())

################################################################################

scores = cross_val_score(model, X_train, y_train, scoring='recall')

print("교차검증 점수(재현율) \n", scores)
print("교차검증 점수(재현율 - 평균) \n", scores.mean())

################################################################################

scores = cross_val_score(model, X_train, y_train, scoring='f1')

print("교차검증 점수(f1) \n", scores)
print("교차검증 점수(f1 - 평균) \n", scores.mean())

################################################################################

model = LogisticRegression(solver='lbfgs',random_state =1).fit(X_train, y_train)
print("학습 결과" , model.score(X_train, y_train))
print("테스트 결과" , model.score(X_test, y_test))

################################################################################

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

