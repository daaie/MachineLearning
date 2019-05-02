# -*- coding: utf-8 -*-

# TensorFlow 모델의 저장
# 1. Meta graph
# - Tensorflow 모델의 graph를 저장하는 파일
# - 모든 Variables, Operations, collections 등을 저장
# - .meta 확장자로 저장됨

# 2. Checkpoint file
# - binary 파일
# - Weights, Biases, Gradients 등이 저장되는 파일
# - Version 0.11부터는 두개의 파일로 저장
#   (model.ckpt.data-00000-of-00001, model.ckpt.index)
# - .data 파일은 Training Variable 을 저장하고 있음
# - checkpoint 파일도 추가로 저장되지만 최근의 상태만을
#   보관하는 파일임(마지막으로 저장된 파일 정보 등)

# TensorFlow 모델의 저장 방법
# tf.train.Saver() 를 활용

import tensorflow as tf

w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

# TensorFlow 모델의 저장하기 위한 객체 생성
saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# TensorFlow 모델을 저장
# 세션 객체를 저장하며, 매개변수로 저장할 경로 및 
# 파일 명을 전달
# saver.save(세션객체, '저장경로 및 파일명', global_step=인덱스정보)
saver.save(sess, './save/model', global_step=100)


















