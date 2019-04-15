# score.csv 파일 읽어서 
# iq academy game tv 점수가 X데이터
# score y를 예측할 수 있는 회기 모델을 테스트하세요
# r2 score 출력

import pandas as pd

####################1. 데이터 파일 불러오기 #############################

fname = '../../data/score.csv'
df = pd.read_csv(fname)

####################불러온 파일 X, y로 나누기 ########################
X_df = df.iloc[:,1:]
y_df = df.iloc[:,0]


# pandas 의 DataFrame 에서 특정 열을 삭제하는 방법
# drop 메소드는 행을 지움. 인덱스를 매개변수로 줘야함.
# axis = 1 을 매개변수로 주면 열을 의미하게됨
# 데이터 프레임 변수명. drop(컬럼명, axis = 1)
# 동작하는 방식 : 매개변수로 전달된 컬럼명의 열을 삭제한 데이터프레임을 반환
# 원본 데이터 프레임은 변환 없음. df는 변환없음. 변수명 선언하고 받아야함.
# 만약 원본데이터 프레임에서 해당열을 삭제하려면 inplace = true를 매개변수로 전달.

df.drop('name', axis = 1)
df.drop('name', axis = 1, inplace = True)

# 특성 데이터의 특징 확인.
print(X_df.info())
print(X_df.describe())
print(X_df.shape)
print(y_df.shape)


X_train = X_df.values
y_train = y_df.values

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

knn_model = KNeighborsRegressor(n_neighbors = 3)
lr_model = LinearRegression()

knn_model.fit(X_train,y_train)
lr_model.fit(X_train,y_train)

print("KNN 학습 모델의 score메소드 :", knn_model.score(X_train,y_train))
print("LR 학습 모델의 score메소드 :", lr_model.score(X_train,y_train))

# 아이큐가 다른 ㄷ데이터에 비해 크기가 큼. 
# 그래서 최근접알고리즘읭 오차가 크다 -> 정규화를 해야함

# 선형 모델은 특성이 많아질 수록 과대 적합됨.
# 선형 모델에서 특성이 많아져도 
# 릿지(l2제약조건-모든가중치를 0에 근사 -> 학습데이터에 안맞게 만드는것 => 테스트데이터에 맞추는것 ) 
# or 랏소(L1 제약조건 - 데이터가 너무 많으면 학습이 느림. 특성데이터 중 몇가지만 유의미하다 . 일부 제외하고 0으로 근사) 
# 를 이용해서 과대적합 안되게 해야함.
# 일반적으론 L2를 더 많이씀.



