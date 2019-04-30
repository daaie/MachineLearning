# -*- coding: utf-8 -*-

# GridSearchCV 클래스를 사용하여
# sms.csv 파일을 분석할 수 있는 최적의 모델을 검색하여
# 분석 결과를 출력하세요.

import pandas as pd

fname = '../../data/sms.csv'
sms = pd.read_csv(fname)

# 특성데이터가 열 하나로 이루어져있기 때문에
# Series 타입으로 처리
X = sms.message
#print(type(X))

y = sms.label
#print(type(y))
#print(y.value_counts() / len(y))

# 특성(X) 데이터의 전처리
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', min_df=2).fit(X.values)

#print("토큰 개수 : ", len(vectorizer.vocabulary_))
#print("변환 결과(희소행렬) : ", vectorizer.transform([X.values[1]]))
#print("변환 결과(밀집행렬) : ", vectorizer.transform([X.values[1]]).toarray())

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X.values, y.values, 
                     stratify=y.values, random_state=0)

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# 모델 객체 생성
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold

model = LogisticRegression(solver='lbfgs', max_iter=10000)

kfold = KFold(n_splits=10, shuffle=True)
grid_model = GridSearchCV(
        model, param_grid=param_grid, cv=kfold, 
        n_jobs=-1, return_train_score=True)

grid_model.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_model.score(X_test, y_test)))
print("최적 매개변수: {}".format(grid_model.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_model.best_score_))
print("최고 성능 모델:\n{}".format(grid_model.best_estimator_))

from sklearn.metrics import classification_report
pred_train = grid_model.predict(X_train)
pred_test = grid_model.predict(X_test)

print("학습 결과 보고서")
print(classification_report(y_train, pred_train))
print("테스트 결과 보고서")
print(classification_report(y_test, pred_test))


















