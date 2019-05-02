# -*- coding: utf-8 -*-

# # TensorFlow 모델의 복원

import tensorflow as tf

# 1. 그래프(네트워크 생성)
# - .meta 파일의 복원
# - tf.train.import() 함수를 이용
# - tf.reset_default_graph() 함수를 사용하여 
#   기존의 그래프를 초기화하는 것이 안전함
tf.reset_default_graph()

saver = tf.train.import_meta_graph('./save/model.meta')

# 2. 파라메터 복원
# - tf.train.Saver()를 이용해서 저장된 모든 파라메터를 복원
with tf.Session() as sess:    
    saver.restore(sess, 
                  tf.train.latest_checkpoint('./save/'))
    print(sess.run('w1:0'))
    print(sess.run('w2:0'))





