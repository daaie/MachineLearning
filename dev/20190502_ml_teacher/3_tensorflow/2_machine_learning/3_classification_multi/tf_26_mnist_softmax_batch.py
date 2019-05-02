# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST/", one_hot=True)

import tensorflow as tf

nFeatures = 784
nClasses = 10

X = tf.placeholder(tf.float32, shape=[None, nFeatures])
y = tf.placeholder(tf.float32, shape=[None, nClasses])

W = tf.Variable(tf.random_uniform([nFeatures, nClasses]))
b = tf.Variable(tf.random_uniform([nClasses]))

h = tf.nn.softmax(tf.matmul(X, W) + b)

loss = tf.reduce_mean(tf.reduce_sum(y * -tf.log(h), axis=1))

train = tf.train.AdamOptimizer().minimize(loss)

predicted = tf.argmax(h, axis=1)
correct = tf.equal(predicted, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

epoches = 50
batch_size = 100
iter_num = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for epoch_step in range(1, epoches+1) :
        
        loss_avg = 0
        for batch_step in range(1, iter_num + 1) :
            batch_X, batch_y = mnist.train.next_batch(batch_size)
            
            _, loss_val = sess.run([train, loss], 
                     feed_dict={X:batch_X, y:batch_y})
            
            loss_avg = loss_avg + loss_val / iter_num
            
        print(f"epoch_{epoch_step} : {loss_avg}")            
        
    loss_val, acc_val = sess.run([loss, accuracy], 
                     feed_dict={X:mnist.test.images,
                                y:mnist.test.labels})

    print(f"TEST : {loss_val}, {acc_val}")











