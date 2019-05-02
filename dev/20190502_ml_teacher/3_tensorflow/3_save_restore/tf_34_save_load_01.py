# -*- coding: utf-8 -*-

import tensorflow as tf

# 기존의 그래프를 초기화(변수 목록 초기화)
tf.reset_default_graph()

w1 = tf.placeholder(tf.float32, name="w1")
w2 = tf.placeholder(tf.float32, name="w2")

b1 = tf.Variable(2.0, dtype=tf.float32, name="bias")

w3 = w1 + w2
w4 = tf.multiply(w3, b1, name="op_to_restore")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

feed_dict = {w1: 4.0, w2: 8.0}
result = sess.run(w4, feed_dict=feed_dict)
print(result)

saver.save(sess, './save/model', global_step=10)











