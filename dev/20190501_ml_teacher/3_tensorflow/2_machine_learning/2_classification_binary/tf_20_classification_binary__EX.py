# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

fname = '../../../data/diabetes.csv'
df = pd.read_csv(fname, header=None)

X_df = df.iloc[:, :-1]
y_df = df.iloc[:,  -1]

print(y_df.value_counts() / len(y_df))

pd.options.display.max_columns = 100

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values,
                     stratify=y_df.values, random_state=1)
    
#리니어 이기 때문에 특성의 개수가 많아야 좋음 -> 특성의 개수를 x의 degree승으로 늘린다.    
#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(degree=3, include_bias=False)
#X_train = poly.fit_transform(X_train)
#X_test = poly.transform(X_test) 
#    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None,164])
y = tf.placeholder(tf.float32, shape=[None])

# 가중치 행렬 텐서
W = tf.Variable(tf.random_normal([164, 1]))

# 절편 텐서
b = tf.Variable(0.0)

# 가설(예측) 식의 정의
# h의 shape는????? -> 2차원 텐서
# None X 30 * 30 X 1 -> None X 1
h = tf.sigmoid(tf.matmul(X, W) + b)

# 행렬 곱의 결과는 2차원 텐서가 반환되므로
# 1차원의 정답 데이터(y)와 연산이 올바르게 처리되지 않습니다.
# 2차원 텐서의 shape를 1차원 텐서로 변경하여
# 오차의 값을 정확히 계산할 수 있도록 함
h_reshape = tf.reshape(h, [-1])

# 손실함수의 정의
loss = tf.reduce_mean(tf.square(        
        y * -tf.log(h_reshape) + (1-y) * -tf.log(1-h_reshape)))

# 학습 객체 선언
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# 예측 값 반환 텐서
predicted = tf.cast(h_reshape >= 0.5, tf.float32)

# 정확도 반환 텐서
accuracy = tf.reduce_mean(tf.cast(
        tf.equal(predicted, y), tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_train, y : y_train}
    
    step = 1
    prev_loss = None    
    prev_w = None
    prev_b = None
    while True :
        sess.run(train, feed_dict=feed_dict)
        
        loss_val, acc_val, w_val, b_val = \
            sess.run([loss, accuracy, W, b], feed_dict=feed_dict)
            
        if step % 100 == 0 :
            print("{0} step : loss -> {1:.2f}, acc -> {2:.2f}".format(
                    step, loss_val, acc_val))
            
        if prev_loss == None or prev_loss > loss_val :
            prev_loss = loss_val
            prev_w = w_val
            prev_b = b_val            
        elif loss_val > prev_loss or np.isnan(loss_val) :
            # 직전의 가중치, 절편의 값으로 복원
            sess.run(tf.assign(W, prev_w))
            sess.run(tf.assign(b, prev_b))
            break
        
        step += 1
        
    feed_dict = {X : X_train, y : y_train}
    loss_val, acc_val = \
            sess.run([loss, accuracy], feed_dict=feed_dict)
    print("학습 결과 : ", loss_val, ", ", acc_val)
    
    feed_dict = {X : X_test, y : y_test}
    loss_val, acc_val = \
            sess.run([loss, accuracy], feed_dict=feed_dict)
    print("테스트 결과 : ", loss_val, ", ", acc_val)


    from sklearn.metrics import confusion_matrix, classification_report
    feed_dict = {X : X_train, y : y_train}
    pred = sess.run(predicted, feed_dict=feed_dict)
    print("학습데이터 confusion matrics")
    print(confusion_matrix(y_train, pred))
    print(classification_report(y_train, pred))
    
    
    feed_dict = {X : X_test}
    pred = sess.run(predicted, feed_dict=feed_dict)
    print("테스트데이터 confusion matrics")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    

from sklearn.svm import SVC

svc = SVC(C =10, gamma=0.01, random_state=1)

svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))





from sklearn.model_selection import train_test_split, KFold, GridSearchCV

param_grid = [{'C' : [1, 0.1, 0.01, 0.001, 10, 100, 1000],
               'gamma' : [1, 0.1, 0.01, 0.001, 10, 100, 1000]}]

print("매개변수 그리드:\n{}".format(param_grid))

kfold = KFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(
        SVC(), param_grid, cv=kfold, 
        return_train_score=True, iid=True, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search.score(X_test, y_test)))

print("최적 매개변수: {}".format(grid_search.best_params_))

print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))










