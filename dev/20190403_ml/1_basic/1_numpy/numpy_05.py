# -*- coding: utf-8 -*-

import numpy as np
 
# numpy 배열은 파이썬 리스트와 같이
# 슬라이싱 연산이 지원됨
python_list = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
              ]

numpy_array = np.array(python_list)

# 시작인덱스 : 종료인덱스.
# 종료인덱스는 포함이 안됨.
# 
slice_1 = numpy_array[0:2, 0:2]
print(slice_1)

# 위와 동일
slice_1 = numpy_array[:2, :2]
print(slice_1)

# 종료인덱스 생략 -> 끝까지
# 시작인덱스 생략 -> 0으로 시작. 
slice_2 = numpy_array[1:, 1:]
print(slice_2)





















