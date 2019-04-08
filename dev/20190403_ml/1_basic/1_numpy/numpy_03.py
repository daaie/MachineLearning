# -*- coding: utf-8 -*-

import numpy as np

# 종료되는 값은 포함 안함. 1~10
python_list = list(range(1,11))

numpy_array_1 = np.array(python_list)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"{numpy_array_1}")


# 배열의 형태를 수정할 수 있는 reshape() 메소드##@@@@#####@@쭝요!@@@###
# -1이면 알아서 값이 들어감.
# 5행 2열이 됨  
numpy_array_2 = numpy_array_1.reshape(-1,2)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"numpy_array_2.shape -> {numpy_array_2.shape}")
print(f"numpy_array_2 -> {numpy_array_2}")

#안나누어지는걸로 나누면 에러남##########@@@@@@@@@@@@@@@@@@@@@#
#numpy_array_2 = numpy_array_1.reshape(-1,3)

numpy_array_3 = numpy_array_2.reshape(-1)
print(f"numpy_array_3.shape -> {numpy_array_3.shape}")
print(f"numpy_array_3 -> {numpy_array_3}")



np.m







