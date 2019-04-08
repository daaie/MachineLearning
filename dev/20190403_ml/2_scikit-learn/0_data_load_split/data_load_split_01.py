# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:08:44 2019

@author: 502-23
"""

import pandas as pd
from sklearn.datasets import load_wine 

# 사이킷 런에서 제공하는 훈련데이터 로딩.
wine = load_wine()

# 분류하는 작업은 타겟과 타겟이름이 제공됨.
# 예측한 결과의 이름을 확인할 수 있도록 target_name제공
# 키가 제공됨.
print(wine.keys());
print(wine.target);
print(wine.target_names);

# 다차원배열은 대문자 이름.
# 일차우너배열은 소문자 이름.
# 특성데이터 
X_df = pd.DataFrame(wine.data)
# 라벨데이터 = 정답.
y_df = pd.DataFrame(wine.target)

# 입력데이터 개수 및 타입. 예측데이터 확인.
print(X_df.info()) 

# 디스ㅡ플레이 생략되니까 조절해줌.
pd.options.display.max_columns = 100
# 입력데이터의 모든 수치값을 확인.
print(X_df.describe())

# 라벨 = 정답 데이터
# 2번 덷이터는 26퍼센트 / 1번데이터 40퍼센트
# 2번데이터 예측하기 힘들다.
print(y_df[0].value_counts())
print(y_df[0].value_counts()/len(y_df))

###################머신러닌ㅇ을 위햐ㅏㄴ 데이터 분할 작업###############
# 학습 ㅈ전 단계
# 1. 데이터 분할
# - 학습 덷이터, 테스트 데이터, 검증 데이터
# - 학습 데이터 : 머신러닝 모델이 학습할 데이터
# - 테스트 데이터 : 학습이 종료된 머신러닝 모델이 정답을 예측하는 데이터.
#(머신러닝 모델의 일반화 정도를 판단하는 기준이 됨 - 학습데이터 100/ 테스트데이터가 100에 가까우면 이상적.)
# - 검증 데이터 : 머신러닝에 사용안함 / 
#                딥러닝 (경사하강법.한발짝식 배우는거)- 텐서플로우에 사용.
#                딥러닝과 같이 한단계실 배울 때 검증하기 위한 데이터. 검증데이터와 학습데이터는 같이 ㅇ성장해야함.
#                학습데이터의 정확도와 검증데이터의 정확도의 추이를 비교하여 학습 과적합 여부를 판단.
#                (데이터는 학습 70%. 테스트 20% 검증 10% 일반적으로 많이 사용)
# 사이킷 런은 이 자르는 라이브러리를 가지고 있다.

# 사이킷 런의 데이터 분할을 위해 제공하는 train_test_split 함수.
from sklearn.model_selection import train_test_split

# train_test_split 함수 사용법
# train_test_split(X,y, 추가적인 파라메터 정보....)
# 추가적인 파라메터 정보에는 
# random_state : 난수의 발생의 seed 값을 의미 (0~1)
# 이 함수는 ㅇ데이터를 이미 다 섞어서 정해준 비율에 맞ㅈ게 잘라줌.
# 그래서 시드값을 줘서 잘라야 동일하게 잘라서 데이터 비교가 가능하게 한다.
# 항상 동일한 데이터 셋이 반환되도록 보장. = 다수번의 학습 시 비교를 수월하게 진행할 수 있음.
# test_size : 테스트 데이터 셋의 비율(실수의 값 사용 ) 
# - 0.3 입력되는 경우 학습 데이터 70 / 테스트 데이터 30 반환됨.
# test_size 지정하지 ㅈ않는 경우 디폴트는 75%/25% 임.

# train_test_split 함수의 반환 값
# X_train(학습할 입력데이터), X_test(테스트할 입력데이터), 
# y_train(학습할 라벨데이터), y_test(테스트할 라벨데이터) 순서대로 반환함.
# = train_test_split (...) 

# train_test_split 사용 예.
# pandas 데이터 프레임에서 numpy 배열을 반환받는 방법.
# values 속성을 사용하여 numpy 배열을 반환받을 수 있음.
# 사이킷 런의 모든 학습을 위한 클래스들은 
##########꼮!!!!! 입력데이터를 2차원 배열, 라벨데이터를 1차원 배열로 함!!!!!!!!.################################################
X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df[0].values, random_state = 1)
# y_df.values 는 (178,1)의 이차원배열임. 그래서 뭔 짓을 해줘야함. 여기선 y_df[0].values로 해쥼.

# 데이터 분할 비율 확인.
print(X_train.shape)
print(X_test.shape)

# 라벨 데이터 확인.
print(y_train.shape)
print(y_test.shape)
print(y_train[:10])
print(y_test[:10])










