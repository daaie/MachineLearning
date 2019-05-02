# -*- coding: utf-8 -*-

import tensorflow as tf

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 텐서플로우 변수의 이름 영역을 지정하는 예제
# 아래와 같이 각 히든계층을 tf.name_scope를
# 사용하여 정의하는 경우 텐서보드에서
# 해당 이름 영역 내부에 있는 변수들을 
# 특정 그룹으로 확인할 수 있습니다.
with tf.name_scope('hidden_1') as scope :
    W1 = tf.Variable(tf.zeros([30, 100]))
    tf.summary.histogram('W1', W1)
    b1 = tf.Variable(tf.zeros([100]))
    tf.summary.histogram('b1', b1)
    
    h1 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('hidden_2') as scope :
    W2 = tf.Variable(tf.zeros([100, 50]))
    tf.summary.histogram('W2', W2)
    b2 = tf.Variable(tf.zeros([50]))
    tf.summary.histogram('b2', b2)
    
    h2 = tf.sigmoid(tf.matmul(h1, W2) + b2)

with tf.name_scope('output') as scope :
    W = tf.Variable(tf.zeros([50, 1]))
    tf.summary.histogram('W', W)
    b = tf.Variable(tf.zeros([1]))
    tf.summary.histogram('b', b)
    
    h = tf.sigmoid(tf.matmul(h2, W) + b)

loss = tf.reduce_mean(y * -tf.log(h) + (1 - y) * -tf.log(1 - h))
tf.summary.scalar('loss', loss)

train = tf.train.AdamOptimizer().minimize(loss)

predicted = tf.cast(h >= 0.5, tf.float32)
accuracy = tf.reduce_mean(
        tf.cast(tf.equal(y, predicted), tf.float32))
tf.summary.scalar('accuracy', accuracy)

init_op = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init_op)
    
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    cancer = load_breast_cancer()
    
    X_df = pd.DataFrame(cancer.data)
    y_df = pd.DataFrame(cancer.target)
    
    X_train, X_test, y_train, y_test = \
                    train_test_split(X_df.values, 
                     y_df.values, random_state=11)

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_fit = scaler.transform(X_train)
    X_test_fit = scaler.transform(X_test)   
    
    merged_summary = tf.summary.merge_all()    
    writer = tf.summary.FileWriter('../logs/tf_43_tensorboard_04')
    writer.add_graph(sess.graph)
    
    for step in range(1,20001) :
        sess.run(train, feed_dict={X:X_train_fit, y:y_train})
        
        if step % 100 == 0 :
            loss_val, summary = sess.run(\
                [loss, merged_summary], feed_dict={X:X_train_fit, y:y_train})
            print(step, loss_val)
            
            writer.add_summary(summary, global_step=step)
            
    writer.close()
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    pred_train = sess.run(predicted, feed_dict={X:X_train_fit})
    print(confusion_matrix(y_train, pred_train))
    print(classification_report(y_train, pred_train))
    
    pred_test = sess.run(predicted, feed_dict={X:X_test_fit})
    print(confusion_matrix(y_test, pred_test))
    print(classification_report(y_test, pred_test)) 
    








