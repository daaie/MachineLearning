# -*- coding: utf-8 -*-

import tensorflow as tf

tf.reset_default_graph()

X_data = [1.,2,3,4,5,6,7,8,9,10]
y_data = [0.,0,0,1,0,1,1,1,1,1]

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.ones([1]))
tf.summary.histogram('w', w)
b = tf.Variable(tf.zeros([1]))
tf.summary.histogram('b', b)

h = tf.sigmoid(X * w + b)

loss = tf.reduce_mean(
        y * -tf.log(h) + (1 - y) * -tf.log(1 - h))
tf.summary.scalar('loss', loss)

train = tf.train.AdamOptimizer().minimize(loss)

predicted = tf.cast(h >= 0.5, tf.float32)
accuracy = tf.reduce_mean(
        tf.cast(tf.equal(y, predicted), tf.float32))
tf.summary.scalar('accuracy', accuracy)

init_op = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init_op)
    
    merged_summary = tf.summary.merge_all()    
    writer = tf.summary.FileWriter('./logs/tf_41_tensorboard_02')    
    writer.add_graph(sess.graph)
    
    for step in range(1,20001) :
        sess.run(train, feed_dict={X:X_data, y:y_data})
        
        if step % 100 == 0 :
            loss_val, summary = sess.run(\
                    [loss, merged_summary], 
                    feed_dict={X:X_data, y:y_data})
            print(step, loss_val)
            
            writer.add_summary(summary, global_step=step)            
    
    writer.close()
    print(sess.run(h, feed_dict={X:X_data}))
    













