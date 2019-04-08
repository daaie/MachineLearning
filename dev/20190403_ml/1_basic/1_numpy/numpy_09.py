# -*- coding: utf-8 -*-

import numpy as np
 
numpy_array_1 = np.array([1,2,3])
numpy_array_2 = np.array([4,5,6])

# 배열 연산 배열 -> 각 요소끼리 연산
# 배열 연산 수 -> 모든 요소에 연산 

# numpy 배열의 각 요소의 합계
numpy_array_r = numpy_array_1 + numpy_array_2
numpy_array_r = numpy_array_1 + 100
numpy_array_r = numpy_array_1 + [100, 200, 300]
numpy_array_r = numpy_array_1 + [[100], [200], [300]]
numpy_array_r = numpy_array_1 + [[100], [200], [300], [400]]
numpy_array_r = numpy_array_1 + [[100,200,300], [200,300,400], [300,400,500], [400,400,400]]
# 타입 에러 numpy_array_r = numpy_array_1 + [100, 200]
# 타입 에러 numpy_array_r = numpy_array_1 + [[100,100], [200,100], [300,100], [400,100]]
numpy_array_r = np.add(numpy_array_1, numpy_array_2)
print(numpy_array_r)

# numpy 배열의 각 요소의 차
numpy_array_r = numpy_array_1 - numpy_array_2
# numpy_array_r = np.subtract(numpy_array_1, numpy_array_2)
print(numpy_array_r)

# numpy 배열의 각 요소의 곱
numpy_array_r = numpy_array_1 * numpy_array_2
# numpy_array_r = np.multiply(numpy_array_1, numpy_array_2)
print(numpy_array_r)

# numpy 배열의 각 요소의 나눗셈
numpy_array_r = numpy_array_1 / numpy_array_2
# numpy_array_r = np.divide(numpy_array_1, numpy_array_2)
print(numpy_array_r)




