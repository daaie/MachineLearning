# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# KNN 알고리즘을 적용하지 않고 예측하는 예제

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
y_train = ['male', 'male', 'male', 'male', 'female', 
           'female', 'female', 'female', 'female']

plt.figure()
plt.title('Human Heights and Weights by Gender')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', 
                marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
K=3
# K는 최근접 이웃의 개수. 인자로 전달함.
model = KNeighborsClassifier(n_neighbors=K)
# 예측기 클래스의 fit 메소드를 사용하여 데이터 학습
# classifier 의 핏은 학습 하지 않고 저장만 함.
model.fit(X_train, y_train)

# 테스트 데이터 x 생성
# 예측에 사용되는 입력데이터의 형태를 2차원으로 지정
X_test= np.array([[155,70]])
predicted = model.predict(X_test)
print("예측 결과 : ", predicted[0])


plt.figure()
plt.title('Human Heights and Weights by Gender')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

# 파이썬은 들여쓰기하면 그냥 포문에 포함됨..괄확 아니라 들여쓰기로 포문을 돌린다.
for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', 
                marker='x' if y_train[i] == 'male' else 'D')


plt.scatter(X_test[:,0], X_test[:,1], c='r', 
                marker='v')
    
plt.grid(True)
plt.show()




X_test = np.array([
        [168, 65],
        [180, 96],
        [160, 52],
        [169, 67]])

print(X_test.shape)
    
predicted = model.predict(X_test)
print("예측 결과 : ", predicted)


plt.figure()
plt.title('Human Heights and Weights by Gender')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

# 파이썬은 들여쓰기하면 그냥 포문에 포함됨..괄확 아니라 들여쓰기로 포문을 돌린다.
for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', 
                marker='x' if y_train[i] == 'male' else 'D')

for i, x in enumerate(X_test):
    plt.scatter(x[0], x[1], c='r', 
                marker='x' if predicted[i] == 'male' else 'D' )
    
plt.grid(True)
plt.show()


# 모델의 평가
# score 메소드 
# 분류 모델인 경우 score 메소드는 정확도를 반환
# 회기 모델인 경우 score 메소드는 R2 점수를 반환
print ("학습 데이터 평가 : ", model.score(X_train, y_train))

predicted_proba = model.predict_proba(X_train[:])
print(predicted_proba)

# 테스트 데이터 셋의 정답(라벨)
y_test = ['male','male','female','female']
print ("테스트 데이터 평가 : ", model.score(X_test, y_test))







