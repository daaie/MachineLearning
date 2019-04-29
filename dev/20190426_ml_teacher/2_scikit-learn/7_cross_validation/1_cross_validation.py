# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 머신러닝을 사용하여 데이터를 분석하는 과정

# 1. 데이터 셋의 로딩
# - 임의의 데이터셋 생성
X, y = make_blobs(random_state=0)

# 2. 데이터의 분할
# - 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=0)
    
# 3. 머신러닝 모델의 학습    
# - 머신러닝 모델 객체의 생성 및 학습
logreg = LogisticRegression(
        solver='liblinear', multi_class='ovr').fit(X_train, y_train)

# 4. 머신러닝 모델의 평가
# - 모델의 평가(테스트 세트 사용)
print("테스트 세트 점수: {:.2f}".format(logreg.score(X_test, y_test)))













