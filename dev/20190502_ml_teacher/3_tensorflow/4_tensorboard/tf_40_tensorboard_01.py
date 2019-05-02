# -*- coding: utf-8 -*-

# TensorBoard : 텐서플로우 모델의 그래프 및 변수들의
# 정보를 시각화하여 보여줄 수 있는 툴
# 텐서플로우 모델을 구성하고 있는 각 변수들의 데이터 흐름을
# 시각화하여 확인할 수 있음
# 추가적으로 모델의 그래프 모습을 확인할 수 있음

# TensorBoard 의 실행방법
# CMD 또는 커맨드 프로그램을 이용하여
# tensorboard --logdir='시각화정보의저장경로'
# 시각화 정보의 저장 경로는 writer 객체에 지정한 경로 

from time import sleep
import tensorflow as tf

# 텐서플로우 그래프 초기화 함수
tf.reset_default_graph()

# 변수 텐서
sum_tensor = tf.Variable(0.0)

# 시각화하여 확인할 변수를 지정
# tf.summary.scalar : 특정 텐서의 값 변화를
# 선의 형태로 시각화할 수 있는 함수
tf.summary.scalar("sum_tensor", sum_tensor)

# 연산텐서
sum_add = tf.assign(sum_tensor, tf.add(sum_tensor, 1.5))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    # tf.summary.merge_all()
    # 현재 모델에 등록된 (tf.summary를 사용하여)
    # 모든 변수의 데이터를 취합하는 객체
    merged_summary = tf.summary.merge_all()
    # 텐서보드를 사용하여 시각화하기 위한 정보를
    # 출력할 수 있는 객체
    writer = tf.summary.FileWriter('./logs/tf_40_tensorboard_01')
    # 현재 텐서플로우 모델을 출력
    writer.add_graph(sess.graph)
    
    for step in range(1, 101) :
        _, sum_val, summary = sess.run([sum_add, sum_tensor, merged_summary])
        print(f"step_{step} : {sum_val}")
        
        # 현재 상태의 값을 보관하고 있는 merged_summary를
        # 사용하여 파일에 출력
        writer.add_summary(summary, global_step=step)
        sleep(0.5)
        
    # 출력 객체 종료
    writer.close()











