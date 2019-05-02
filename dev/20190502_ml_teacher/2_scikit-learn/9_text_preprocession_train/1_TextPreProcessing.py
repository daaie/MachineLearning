# -*- coding: utf-8 -*-

# 머신러닝에서 문자 형태로 저장된 특성데이터를 
# 사용하여 학습하는 방법

# 1. 문자형의 데이터를 수치데이터로 변경
# - 문자열로 구성된 각 단어를 구분하여 라벨 값을 지정
# - 문자열을 구성하고 있는 각 단어의 빈도수를 계산
# - 문자열의 형태를 각 단여의 빈도수로 변경


# DictVectorizer 클래스
# 문자열을 구성하는 각 단어의 수를 세어놓은 
# 딕셔너리 타입에서 BOW 벡터를 생성
# BOW : Bag Of Words

from sklearn.feature_extraction import DictVectorizer

dicts = [{'A':1, 'B':2}, {'B':3, 'C':1, 'D':2}]

vectorizer = DictVectorizer(sparse=False)

dicts_transform = vectorizer.fit_transform(dicts)

print("변환 결과 : ", dicts_transform)

print("구성 단어의 이름 : ", vectorizer.feature_names_)

print("변환 결과 : ", vectorizer.transform({'A':2, 'D':1}))

# DictVectorizer 클래스의 학습에 사용되지 않은 단어는
# 변환 과정에서 처리되지 않습니다.
# 아래의 예에서 F는 변환 결과에 포함되지 않는 것을 확인할 수 있습니다.
print("변환 결과 : ", vectorizer.transform({'A':2, 'B':1, 'F':3}))
















