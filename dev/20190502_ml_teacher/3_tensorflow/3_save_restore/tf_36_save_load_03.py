# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST/", one_hot=True)

import tensorflow as tf

tf.reset_default_graph()

nFeatures = 784
nClasses = 10

X = tf.placeholder(tf.float32, shape=[None, nFeatures])
y = tf.placeholder(tf.float32, shape=[None, nClasses])

# 입력데이터를 압축하는 계층을 추가
W_hidden_1 = tf.Variable(tf.random_uniform([nFeatures, 392]))
b_hidden_1 = tf.Variable(tf.random_uniform([392]))

# 분류 작업의 경우 숨겨진 계층은 활성화 함수의 
# 결과를 반환합니다.
h_hidden_1 = tf.sigmoid(
        tf.matmul(X, W_hidden_1) + b_hidden_1)

W = tf.Variable(tf.random_uniform([392, nClasses]))
b = tf.Variable(tf.random_uniform([nClasses]))

h = tf.nn.softmax(tf.matmul(h_hidden_1, W) + b)

loss = tf.reduce_mean(tf.reduce_sum(y * -tf.log(h), axis=1))

train = tf.train.AdamOptimizer().minimize(loss)
#train = tf.train.GradientDescentOptimizer(
#        learning_rate=0.1).minimize(loss)

predicted = tf.argmax(h, axis=1)
correct = tf.equal(predicted, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

epoches = 20
batch_size = 100
iter_num = int(mnist.train.num_examples / batch_size)

import os
import sys
# tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
SAVER_DIR = "save"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "mnist")
# save 디렉토리에 저장된 checkpoint 파일을 확인하여
# 저장된 파일이 있는지 확인
# (저장된 파일이 존재하는 경우 마지막에 저장된 파일명이 반환)
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    # 만약 저장된 모델과 파라미터가 있으면 이를 불러오고 (Restore)
    # Restored 모델을 이용해서 테스트 데이터에 대한 정확도를 출력하고 프로그램을 종료합니다.
    if ckpt and ckpt.model_checkpoint_path:    
        saver.restore(sess, ckpt.model_checkpoint_path)    
        print("테스트 데이터 정확도 (Restored) : %f" % accuracy.eval(feed_dict={X:mnist.test.images,
                                y:mnist.test.labels}))
        sys.exit(0)
    
    for epoch_step in range(1, epoches+1) :
        
        loss_avg = 0
        for batch_step in range(1, iter_num + 1) :
            batch_X, batch_y = mnist.train.next_batch(batch_size)
            
            _, loss_val = sess.run([train, loss], 
                     feed_dict={X:batch_X, y:batch_y})
            
            loss_avg = loss_avg + loss_val / iter_num
        
        saver.save(sess, checkpoint_path, global_step=epoch_step)
        print(f"epoch_{epoch_step} : {loss_avg}")   

    loss_val, acc_val = sess.run([loss, accuracy], 
                     feed_dict={X:mnist.train.images,
                                y:mnist.train.labels})

    print(f"TRAIN : {loss_val}, {acc_val}")         
    
    loss_val, acc_val = sess.run([loss, accuracy], 
                     feed_dict={X:mnist.test.images,
                                y:mnist.test.labels})

    print(f"TEST : {loss_val}, {acc_val}")











