# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:44:06 2019

@author: 502-23
"""

#pip list
#없으면 홈페이지간다
#https://www.tensorflow.org/
#install tab -> goto pip guide

# 각각의 텐서들을 엣지를 통해 이은 것.
# 싸이킷 런은 절차지향이었다면 텐서플로우는 객체지향
# 정의를 먼저써줌. 7+3 이면 7텐서 하나 +텐서 하나 3텐서하나 ->3개의 텐서 
# 세션의 생성 
# 세션을 통해서 그래프, 수치 조정 등 .

import tensorflow as tf
print(f"tensorflow's verison -> {tf.__version__}")

msg = tf.constant(7)

msg2 = tf.constant(3)
# 텐서플로우 전용 변수 
sess = tf.Session()
sess.run()