# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:02:35 2019

@author: 502-23
"""

# -*- coding: utf-8 -*-

import pandas as pd

# 파일의 경로를 포함한 이름을 저장
fname = '../../data/winequality-red.csv'
# 특정 파일을 DataFrame으로 로딩
df = pd.read_csv(fname, sep=';')

# iris 데이터의 일부분을 확인
print(df)
print(df.head())
print(df.tail())

print(df.info())
print(df.describe())

# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 추출하는 코드
# iloc[행, 열]
X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

# 라벨 데이터의 분포를 확인
print(type(X_df))
print(X_df)

# y는 시리즈 타입으로 분할해줌. 타입으로 확인가능 type()
print(type(y_df))
print(y_df.shape)

print(type(y_df))
# 5,6빼고 다 맞추기힣ㅁ듬. 편향된 데이터 
print(y_df.value_counts())
print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, random_state = 1)

print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))


print(X_train.shape)
print(X_test.shape)

# 라벨 데이터 확인.
print(y_train.shape)
print(y_test.shape)

print(y_train[:10])
print(y_test[:10])
