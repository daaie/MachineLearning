# -*- coding: utf-8 -*-

# 머신러닝에서 문자 형태로 저장된 특성 데이터를 
# 사용하여 학습하는 방법

# 1. 문자형의 데이터를 수치데이터로 변경
# - 문자열로 구성된 각 단어를 구분하여 라벨 값을 지정
# - 문자열을 구성하고 있는 각 단어의 빈도수를 계산
# - 문자열의 형태를 각 단여의 빈도수로 변경

# 2. 각 문서(말뭉치를 구성하는 각 샘플 데이터)를 토큰 개수
# 만큼의 행렬로 표현
# - 각 문서를 구성하고 있는 토큰의 개수를 카운팅하여
# - 0 또는 개수를 출력
# (문서 데이터를 서로 다른 길이를 가지므로, 모든 문서데이터가
# 동일한 크기의 특성을 가지도록 강제하는 방법)

# CountVectorizer 클래스 
# 문서(하나 또는 다수개의 텍스트 문장) 집합에서
# 단어들의 토큰을 생성하고, 각 단어의 수를 카운팅하여
# BOW 타입으로 인코딩 된 벡터를 생성하는 클래스
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Hello Python', 
          'This is the first document',
          'This is the second document',
          'And thr third document',
          'is this the first document?',
          'The last document']

vectorizer = CountVectorizer()
vectorizer.fit(corpus)

print("토큰의 개수 : ", len(vectorizer.vocabulary_))
print("토큰의 내용 : ", vectorizer.vocabulary_)

print("변환 결과(희소행렬) : ", 
      vectorizer.transform(['This is the second document']))
print("변환 결과(밀집행렬) : ", 
      vectorizer.transform(['This is the second document']).toarray())

# 다수개의 문서를 입력받아
# 각 문서를 BOW 인코딩 벡터로 변환하는 예제
print("변환 결과(다수개의 문저) : \n", vectorizer.transform(corpus).toarray())





















