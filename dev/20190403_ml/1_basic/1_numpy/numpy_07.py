# -*- coding: utf-8 -*-

import numpy as np
 
# numpy bool 인덱싱
# 배열 각 요소의 선택 여부를 True, False로 표현하는 방식
python_list = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

numpy_array = np.array(python_list)
 
bool_indexing_array = np.array([
    [False,  True, False],
    [True, False,  True],
    [False,  True, False]
])

# 일차원 반환.
bool_indexing_result = numpy_array[bool_indexing_array];
print(bool_indexing_result)    







