# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../data/sms.csv'
sms = pd.read_csv(fname)

# 특성데이터가 열 하나로 이루어져있기 때문에
# Series 타입으로 처리
X = sms.message
print(type(X))

y = sms.label
print(type(y))
print(y.value_counts() / len(y))

# 특성(X) 데이터의 전처리
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer().fit(X.values)

print("토큰 개수 : ", len(vectorizer.vocabulary_))
print("변환 결과(희소행렬) : ", vectorizer.transform([X.values[1]]))
print("변환 결과(밀집행렬) : ", vectorizer.transform([X.values[1]]).toarray())

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X.values, y.values, 
                     stratify=y.values, random_state=0)

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')

from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X_train, y_train, cv=kfold)

print("교차검증 점수 : \n", scores)
print("교차검증 점수 : \n", scores.mean())

# 모델 객체 생성
model = LogisticRegression(solver='lbfgs', random_state=1).fit(X_train, y_train)
print("학습 결과 : ", model.score(X_train, y_train))
print("테스트 결과 : ", model.score(X_test, y_test))

from sklearn.metrics import classification_report
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print("학습 결과 보고서")
print(classification_report(y_train, pred_train))
print("테스트 결과 보고서")
print(classification_report(y_test, pred_test))


















