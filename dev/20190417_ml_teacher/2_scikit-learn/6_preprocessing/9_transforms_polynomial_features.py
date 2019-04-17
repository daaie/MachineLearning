# -*- coding: utf-8 -*-

# 데이터전처리
# 데이터 분석을 위한 데이터 처리 과정
# - 전체 데이터 셋에서 데이터 분석에 사용될 열 선정
# - 특정 열에 존재하는 빈 값을 제거하거나
#   또는 특정 열에 존재하는 빈 값을 임의의 값으로 변경
# - 데이터의 스케일(값의 범위) 조정
# - 범주형 변수의 값 변경
#   (문자열 값의 수치 데이터화)
#   (원핫인코딩 처리)
# - 학습, 테스트 데이터 분할

# 특성 데이터의 확장 - 다항식 변환
# - PolynomialFeatures 클래스
# 방정식의 차원을 높여줌. 1차방정식의 형태를->n차 방정식의 형태로 
# - 입력값  x 를 다항식으로 변환하는 기능을 제공

# 파라메터 정보
# degree : 다항식의 차수
# include_bias : 상수항(절편) 생성 여부

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(-1, 1)
print(X)

pf = PolynomialFeatures(degree=2)
#pf = PolynomialFeatures(degree=2, include_bias=False)
print(pf.fit_transform(X)) # [절편, x, x^2]


# 날씨같은 데이터를 예측할때는 다차원이어야함.
# 차수를 올리는 것은 정형화와 상관이없다..
# 보통 회기분석을 위한 것.

# fit 은 어쨋든 학습데이터에만 해야함.