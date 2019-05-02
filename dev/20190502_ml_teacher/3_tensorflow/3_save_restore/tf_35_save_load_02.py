# -*- coding: utf-8 -*-

import tensorflow as tf

tf.reset_default_graph()

sess=tf.Session()    

saver = tf.train.import_meta_graph('./save/model-10.meta')
saver.restore(sess, tf.train.latest_checkpoint('./save/'))

# 모델의 변수 텐서 복원
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
b1 = graph.get_tensor_by_name("bias:0")

# 모델의 연산 텐서 복원
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# 사용할 수 있는 모든 텐서 확인 코드
for op in tf.get_default_graph().get_operations():
    print(op.name)

feed_dict ={w1:13.0,w2:17.0}
print (sess.run(op_to_restore, feed_dict=feed_dict))
print (sess.run(b1))















