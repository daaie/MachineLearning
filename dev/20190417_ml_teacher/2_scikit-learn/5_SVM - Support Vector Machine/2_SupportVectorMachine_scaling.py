# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# 서포트벡터머신은 스케일에 많은 영향을 받는다
# 스케일이 비슷해야 분할을 잘함.

# 특성 데이터의 스케일이 서로 다른 영역에 위치한 경우
# 공간의 분할이 어려워 지므로 학습이 올바르게 진행되지 않습니다.
# SVM알고리즘 기반의 예측기를 사용하는 경우
# 반드시 데이터의 전처리를 수행해야만 올바르게 학습이 이뤄집니다.




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




from sklearn.svm import SVC

# SVC 클래스의 하이퍼 파라메터 C - l2제약조건 정규화와 관련된 제약조건.
# 모델의 학습에 제약을 설저어하기 위한 파라메터
# C의 값이 낮을 수록 강한 제약이 설정됨. = 모델이 과적합 된 경우 일반화 성능을 높이기 위해 사용.
# C의 값이 높을 수록 약한 제약이 설정됨. = 모델이 과소적합된 경우 학습 성능을 높이기 위해서 사용.

svc = SVC(gamma=1, random_state = 0, C=10)
svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))

########################################################################
########################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!KNN / SVM -> 전처리 반드씨!!!!!!!!!!!!!!!꼭!!!!!!!!!!!!!!!!!!!!!!필요
########################################################################
########################################################################

# 대다수의 선형모델 -> 전처리 필요
# ..선형회기분석은 가중치를 부여하기 때문에 전처리가 성능으 ㄹ올리진않는다. 

