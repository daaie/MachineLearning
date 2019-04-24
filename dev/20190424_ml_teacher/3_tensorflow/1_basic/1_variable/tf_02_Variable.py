# -*- coding: utf-8 -*-

# 텐서플로우의 텐서 타입
# 1. 상수(constant)
# 2. 변수(Variable)
# 3. 실행매개변수(placeHolder)

import tensorflow as tf

# 텐서플로우의 변수 텐서 선언
# tf.Variable을 사용하여 정의할 수 있음
# 프로그램의 진행 중 값이 변경될 수 있는 텐서
# 변수 텐서를 사용하는 경우 반드시 세션을 통해 초기화를 진행해야만
# 에러가 발생하지 않음
var_1 = tf.Variable(1)

# var_1 텐서의 값을 11 로 수정할 수 있는 연산텐서
# 세션에 의해서 run(var_1_assign) 이 호출되야만
# 실행되는 코드
var_1_assign = var_1.assign(11)

# 파이썬코드로 아래와같음.
# n1 = 1;
# def n1_assign():
#   n1 = 11;
# 누군가가 n1_assing()을 불러야 변경된다는 의미.
# 걍함수를만든것.

var_2 = tf.Variable([1,2,3])

# tf.fill 함수
# 형태, 값을 입력받아 해당 형태의 텐서를 생성하고, 
# 특정 값으로 초기화하는 함수
var_3 = tf.Variable(tf.fill([3,3], 3))

var_4 = tf.Variable([1,2,3,4], dtype=tf.float64)

sess = tf.Session()

# tf.Variable 통해서 선언된 텐서가 단 하나라도 존재하는****!!!!
# 경우 반드시 아래와 초기화 과정을 실행해야만 합니다.->global_variables_initializer()얘를 무조건실행해야함.
init_variables = tf.global_variables_initializer()
sess.run(init_variables)

result = sess.run(var_1)
print(f"result = {result}")

# var_1 의 값을 수정하는 텐서를 실행 assing은 정의된 함수 일 뿐임.
sess.run(var_1_assign)

result = sess.run(var_1)
print(f"result = {result}")

result = sess.run(var_2)
print(f"result = {result}")

result = sess.run(var_3)
print(f"result = {result}")

result = sess.run(var_4)
print(f"result = {result[1]}")

sess.close()