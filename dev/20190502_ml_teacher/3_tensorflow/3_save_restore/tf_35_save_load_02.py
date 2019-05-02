# -*- coding: utf-8 -*-

import tensorflow as tf

# 텐서플로우 그래프 초기화(텐서의 이동 충돌을 방지)
tf.reset_default_graph()

sess=tf.Session()    

saver = tf.train.import_meta_graph('../save/model-10.meta')
saver.restore(sess, tf.train.latest_checkpoint('../save/'))

# 현재 세션이 보관중인 계산 그래프의 객체를 반환받는 코드
# 현재 세션 내부의 모든 텐서 정보를 저장하고 있는 객체 
# 모델의 변수 텐서 복원
graph = tf.get_default_graph()
# w1:0을 하면 0번째 w1이란 소리임. 파이썬에서 1,2,3,4 이렇게 인덱스를 주면서 생성됨
# 근데 tf.get_default_graph() 부르면 그냥 항상 0번째거 만들 수 있음.
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
b1 = graph.get_tensor_by_name("bias:0")

# 모델의 연산 텐서 복원
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print("bias:",sess.run(b1))
#print("op_to_restore:",sess.run(op_to_restore))
# 사용할 수 있는 모든 텐서 확인 코드
for op in tf.get_default_graph().get_operations():
    print(op.name)

feed_dict ={w1:13.0,w2:17.0}
print (sess.run(op_to_restore, feed_dict=feed_dict))
print (sess.run(b1))















