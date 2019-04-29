# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:47:06 2019

@author: 502-23
"""

X_data=[1,2,3,4,5,6,7,8,9,10]
y_data=[0,0,0,0,0,1,1,1,1,1]

import tensorflow as tf
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.ones(shape=[1]))
b = tf.Variable(tf.ones(shape=[1]))

h = X * w + b
# h가 0또는 1이 나와야함.
# 텐서플로우를 활용하여 이진 데이터를 분류하는 ㄴ방법
# 1. 선혛ㅇ방정식을 사용하여 값응ㄹ 예측
# 2. 예측한 값을 활성화 함수를 사용하여 지정된 영역 내부로 값을 압축
# 3. 활성화 함수별로 지정된 기준값을 사용하여 분류값을 예측
pre_h = X*w +b
h = tf.sigmoid(pre_h)
#sigmoid 함수의 실행결과를 활용하여
# 이진 불류의 값을 반환하는 텐서 변수 선언

# 양성데이터(1인경우)의 오차를 계산
# -sigmoid 함수의 실행 결과각 1에 가까ㅜ어 질수록
# 오차의 값을 작게 측정하기 위해서 1-h를 사용하여 
# tf.log 의 결과를 사용함
loss_1 = y * -tf.log(h)


# 양성데이터(0인경우)의 오차를 계산
# -sigmoid 함수의 실행 결과각0에 가까ㅜ어 질수록
# 오차의 값을 작게 측정하기 위해서
# tf.log 의 결과를 사용함
loss_0 = (1-y) * -tf.log(1-h)

loss_sum = loss_0 + loss_1
# 오차를 제곱한 후 평균값을 계산
loss = tf.reduce_min(tf.square(loss_sum))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer()

# 7. OPTIMIZER 객체를 사용하여 학습을 진행시키기 위한
# 텐서 선언
train = optimizer.minimize(loss)

predicted = h>=0.5
predicted_cast = tf.cast(predicted, tf.int32)


accuracy = tf.reduce_min(tf.cast(predicted_cast == h, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X:X_data, y:y_data}
    
    print(sess.run(pre_h, feed_dict=feed_dict))
    print(sess.run(h, feed_dict=feed_dict))
    print(sess.run(predicted, feed_dict=feed_dict))
    print(sess.run(predicted_cast, feed_dict=feed_dict))
    
    for step in range(1,101):
        sess.run(train, feed_dict=feed_dict)
        
        if step%10 ==0:
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict=feed_dict)
            
            print("{0} loss {0:.2f}, acc:{2:.2f}".format(step, loss_val, acc_val))
    