# -*- coding: utf-8 -*-

import tensorflow as tf

tf.reset_default_graph()

sess = tf.Session()    

saver = tf.train.import_meta_graph('../save_mnist/mnist-20.meta')
saver.restore(sess, tf.train.latest_checkpoint('../save_mnist/'))

graph = tf.get_default_graph()

#for op in tf.get_default_graph().get_operations():
#    print(op.name)

## 모델의 변수 텐서 복원
X = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("y:0")

# 모델의 연산 텐서 복원
loss = graph.get_tensor_by_name("loss:0")
accuracy = graph.get_tensor_by_name("accuracy:0")

# 입력 데이터 처리
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../../data/MNIST/", one_hot=True)

# 필요한 텐서만 가지고ㅗ아서 호출.
feed_dict={X:mnist.test.images, y:mnist.test.labels}
loss_val, acc_val = sess.run([loss, accuracy], 
                     feed_dict=feed_dict)
print(f"TEST : {loss_val}, {acc_val}")
















