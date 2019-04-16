
import pandas as pd

####################1. 데이터 파일 불러오기 #############################

fname = '../../data/score.csv'
df = pd.read_csv(fname)

df.drop('name', axis=1, inplace = True)

####################불러온 파일 X, y로 나누기 ###########################
X_df = df.iloc[:,1:]
y_df = df.iloc[:,0]


# 특성 데이터의 특징 확인. ##############################################
print(X_df.info())
print(X_df.describe())
print(X_df.shape)
print(y_df.shape)

######학습 ##########################################################
X_train = X_df.values
y_train = y_df.values

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
# Ridge 클래스는 회기 분석
# RidgeClassifier 분류 분석

knn_model = KNeighborsRegressor(n_neighbors = 3).fit(X_train,y_train)
lr_model = LinearRegression().fit(X_train,y_train)

###Ridge 클래스의 하이퍼 파라메터 alpha####################################
# alpha 가 작을 수록 모든 특성의 가중치(weight) 이 분산되기 때문에 학습데이터의 스코어가 증가함.
# alpha 가 클수록 모든 특성의 가중치(weight)이 0에 근사되기 때문에 학습데이터의 스코어가 감소함.
# alpha가 작아질수록 LR에 가까워짐. -> Alpha가 커질 수록 학습데이터에 최적화되는것을 방지.
# alpha가 클스록 제약이 커지고 작아질수록 제약이 작아짐.
ridge_model = Ridge(alpha=5).fit(X_train,y_train)

print("KNN 학습 모델의 score메소드 :", knn_model.score(X_train,y_train))
print("LR 학습 모델의 score메소드 :", lr_model.score(X_train,y_train))
print("Ridge 학습 모델의 score메소드 :", ridge_model.score(X_train,y_train))

# 학습데이터 스코어기 때문에 Ridge가 LR보다 떨어지는게 일반적. 

# 선형 모델에 L2 제약 조건을 추가한 Ridge 클래스
# L2 제약 조건 : 모든 특성에 대한 가중치의 값을
# 0 주변으로 위치하도록 제어하는 제약조건
# LinearRegression 클래스는 학습 데이터에 최적화되도록
# 학습을 하기때문에 테스트 데이터에 대한 일반화 성능이 감소됩니다.
# 이러한 경우 모든 특정 데이터를 적절히 활용할 수 있도록
# L2 제약 조건을 사용할 수 있으며, L2 제약조건으로 인하여
# 모델의 일반화 성능이 증가하게 됩니다.




###############그래프를 그려보아요###########################################
from matplotlib import pyplot as plt
coef_range = list(range (1, len(ridge_model.coef_) + 1))
plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')

plt.hlines(0,1,len(ridge_model.coef_) + 1,
           colors = 'y', linestyles = 'dashed')

plt.show()
# alpha를 키워가면서 확인해보아랏 .
# 선형모델이 너무 학습데이터만 잘맞추면 제약조건을 준다.


























