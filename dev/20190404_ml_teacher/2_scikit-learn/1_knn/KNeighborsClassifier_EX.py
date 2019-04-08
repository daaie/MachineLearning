# -*- coding: utf-8 -*-

# 1. data 디렉토리에 저자오딘 diabetes.csv 파일의 데이터를 
# 2. KNN 알고리즘을 사용하여 분석한 후
# 3. 정확도를 출력하세요.

# diabetes 데이터느 ㄴ정규화가 되어있음. 최근접 알고리즘으 ㄹ위한 데이터.

import pandas as pd

####################1. 데이터 파일 불러오기 #############################
# 데이터 내부에 헤더 정보가 존재하지 않는 경우.
# 타이틀이 없으면 csv에서 첫번째 데이터가 빠져버릴 수 이따...ㅋㅋㅋㅋㅋ
fname = '../../data/diabetes.csv'
# 불러올 때 잘 불러와야한다.
# header = None 이면 헤더 없이 데이터만 로딩.
df = pd.read_csv(fname, header=None)

####################불러온 파일 X, y로 나누기 ########################
X_df = df.iloc[:,:-1]
y_df = df.iloc[:, -1]

# 특성 데이터의 특징 확인.
print(X_df.info())
print(X_df.describe())
print(X_df.shape)
print(y_df.shape)

####################데이터 분포 확인##################################
# 라벨데이터 비율 확인. 
# 1이 60퍼센트가 넘음.
# 맞다고 해도 60퍼센트임ㅋㅋㅋ정확도가 떨어질수이따.
print(y_df.value_counts())
print(y_df.value_counts() / len(y_df))


####################데이터 분할#######################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, random_state=1)

print("len(X_train) : ", len(X_train))
print("len(X_test) : ", len(X_test))

##########데이터 전처리 과정!!!!!!!!!!!!!!?!!!!!!!!!!!!!!!!!!

#################### 2.KNN예측기로 모델 만들기###########################
from sklearn.neighbors import KNeighborsClassifier
K=3
# 사이킷 런의 모든 예측기는 fit 메소드 실행 시 자신의 객체를 반환.
# 한줄에 쓸수도 이따^__^.
model = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)

#################### 테스트 데이터로 예측결과 얻기######################
predicted = model.predict(X_test)
print("예측 결과 : ", predicted)
#print("예측 결과 퍼센트로 : ",model.predict_proba(X_train[:]))
#print ("테스트 데이터 예측 퍼센트로: ", model.predict_proba(X_test))

############### 3. 정확도 출력#########################################
#################### 학습데이터 평가###################################
print ("학습 데이터 평가 : ", model.score(X_train, y_train))

#################### 테스트 데이터 평가################################
print ("테스트 데이터 평가 : ", model.score(X_test, y_test))



# 최근접알고리즘은 그리드 서치를 이용해서 최적화된 K를 찾아야한다.
# ㄷ최근접 데이터가 많아질수록 일반화됨. ( k가 커질 수록)

train_accuracy = []
test_accuracy = []
K_list = list(range(1, 11, 2))

for k_value in K_list :
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)
    train_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))


from matplotlib import pyplot as plt
plt.title('Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')

plt.plot(list(range(1,len(train_accuracy)+1)), train_accuracy, 'ro')
plt.plot(list(range(1,len(train_accuracy)+1)), train_accuracy, 'r-')
plt.plot(list(range(1,len(test_accuracy)+1)), test_accuracy, 'b--')
plt.plot(list(range(1,len(test_accuracy)+1)), test_accuracy, 'bo')
plt.show()


# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(max_depth = 5).fit(X_train, y_train)
# 가장 좋은 성적의 모델을 생성.
K=3
model = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
#################### 학습데이터 평가###################################
print ("학습 데이터 평가 : ", model.score(X_train, y_train))
#################### 테스트 데이터 평가################################
print ("테스트 데이터 평가 : ", model.score(X_test, y_test))
print(y_df.value_counts() / len(y_df))
# 테스트 score 74 % 근데 1인 데이터 65퍼센트있음.
# 뭐 거의 1찍어도 잘맞춤
# 그래서 스코어를 믿으면 안된다.
# 분류 모델의 성능 평가는 
# 정밀도 / 재현율

from sklearn.metrics import precision_score, recall_score
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
# 정밀도..................양성데이터(1)를 예측한 모든 케이스 중
# 실제로 정답이었던 비율.
# precision_score (정답데이터 ,예측데이터)
print("정밀도(train) : ", precision_score(y_train,pred_train))
print("정밀도(test) : ", precision_score(y_test,pred_test))

# 전체 양성데이터(1) 중 정답을 에측한 비율.
print("재현율(train) : ", recall_score(y_train,pred_train))
print("재현율(test) : ", recall_score(y_test,pred_test))


from sklearn.metrics import confusion_matrix
print("confusion_matricx - train: ", confusion_matrix(y_train,pred_train))
print("confusion_matricx - test: ", confusion_matrix(y_test,pred_test))

from sklearn.metrics import classification_report
print("confusion_matricx - train: ", classification_report(y_train,pred_train))
print("confusion_matricx - test: ", classification_report(y_test,pred_test))

# confusion matricx 테스트 데이터를 보면
# 0을 맞추는 확률은 거의 반타작. 0을 맞춰야하는 경우에는
# 이 모델을 쓰면안됨.





import matplotlib.pyplot as plt
plt.figure()
plt.xlabel('X[1]')
plt.ylabel('X[2]')

# 파이썬은 들여쓰기하면 그냥 포문에 포함됨..괄확 아니라 들여쓰기로 포문을 돌린다.
for i, x in enumerate(X_train):
    plt.scatter(x[1], x[2], c='k', 
                marker='x' if y_train[i] == 0 else 'D')

for i, x in enumerate(X_test):
    plt.scatter(x[1], x[2], c='r', 
                marker='x' if predicted[i] == 0 else 'D' )
    
plt.grid(True)
plt.show()



