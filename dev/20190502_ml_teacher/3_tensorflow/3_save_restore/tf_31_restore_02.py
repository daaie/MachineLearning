# -*- coding: utf-8 -*-

# # TensorFlow 모델의 복원

import tensorflow as tf

# 1. 그래프를 저장한 파일이 존재하지 않는 경우의 처리 방법
# - tf.reset_default_graph() 함수를 사용하여 
#   기존의 그래프를 초기화
# - 기존 모델의 모든 변수를 선언
tf.reset_default_graph()

w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

# 2. 파라메터 복원
# - tf.train.Saver()를 이용해서 저장된 모든 파라메터를 복원
with tf.Session() as sess:    
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./save/'))
    print(sess.run(w1))
    print(sess.run(w2))


