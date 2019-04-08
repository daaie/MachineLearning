# -*- coding: utf-8 -*-

# numpy는 과학 계산을 위한 라이브러리
# 다차원 배열을 처리하는데 필요한 여러 유용한 기능을 제공

# 설치 명령
# pip install numpy

import numpy as np

# numpy 배열 생성 방법
# 파이썬의 리스트를 활용한 생성
# np.array(파이썬 리스트/변수)

python_list = [1, 2, 3, 4]
np_array_1 = np.array(python_list)
# print 않에 f는 포멧팅을 의미. {} 는 변수 출력
print(f"np_array_1 -> {np_array_1}")
print(f"type(np_array_1) -> {type(np_array_1)}")

#(4,)는 1차원 배열 4개이다. 머신러닝에서 shpae는 중요하당. 
print(f"np_array_1.shape -> {np_array_1.shape}")

# 시작인덱스는 0
print(f"np_array_1[0] -> {np_array_1[0]}")
# -는 뒤에서부터 세는거 
print(f"np_array_1[-1] -> {np_array_1[-1]}")
print(f"np_array_1[-2] -> {np_array_1[-2]}")

print(f"len(np_array_1) -> {len(np_array_1)}")
#shape의 [0]은 크기임.
print(f"np_array_1.shape[0] -> {np_array_1.shape[0]}")
 
np_array_2 = np.array([[1,2,3],[4,5,6]])
print(f"np_array_2 -> {np_array_2}")
print(f"type(np_array_2) -> {type(np_array_2)}")
print(f"np_array_2.shape -> {np_array_2.shape}")

#numpy 다차원 배열 요소 [행.열] 
##############################쉼표로 접근한다!##########################
print(f"np_array_2[0,0] -> {np_array_2[0,0]}")
print(f"np_array_2[-1,-1] -> {np_array_2[-1,-1]}")

#이차원 배열 len함수는 행의 값
print(f"len(np_array_2) -> {len(np_array_2)}")
#이차원배열 첫번쨰 요소의 길이는 열의 값.
print(f"len(np_array_2[0]) -> {len(np_array_2[0])}")
#shpae 속성의 값을 사용하여 길이를 반환.


##############################shape 자주 씀!##########################
print(f"np_array_2.shape[0] -> {np_array_2.shape[0]}")
print(f"np_array_2.shape[1] -> {np_array_2.shape[1]}")


























