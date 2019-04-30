# -*- coding: utf-8 -*-

# data 디렉토리에 저장된 sms.csv 파일을 분석하여
# 결과를 확인하세요
# (말뭉치 변환에 TfidfVectorizer 클래스를 활용하세요.)

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
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', min_df=3).fit(X.values)

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

print("교차검증 점수(정확도) : \n", scores)
print("교차검증 점수(정확도 - 평균) : \n", scores.mean())

scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='precision')

print("교차검증 점수(정밀도) : \n", scores)
print("교차검증 점수(정밀도 - 평균) : \n", scores.mean())

scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall')

print("교차검증 점수(재현율) : \n", scores)
print("교차검증 점수(재현율 - 평균) : \n", scores.mean())

scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')

print("교차검증 점수(F1) : \n", scores)
print("교차검증 점수(F1 - 평균) : \n", scores.mean())

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


















