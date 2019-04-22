# -*- coding: utf-8 -*-

# 머신러닝에서 문자 형태로 저장된 특성데이터를 
# 사용하여 학습하는 방법

# 2. 각 문서(말뭉치를 구성하는 각 샘플 데이터)를 토큰 개수
# 만큼의 행렬로 표현
# - 각 문서를 구성하고 있는 토큰의 개수를 카운팅하여
# - 0 또는 개수를 출력
# 문서 데이터를 서로다른 길이를 가지므로 모든 문서데이터가ㅓ 동일한 크기의 특성을 가지도록 강제핳는 방법.


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

# min_df 하이퍼파라메터
# 토큰을 추출할 때 제약조건을 설정하는 파라메터
# 각 토큰은 min_df에 지정 된 개수 이상이 사용되어야만 
# 토큰으로 등록됩니다.
# min_df=2 2번 이상 나온애들이 의미있는단어다 - 토큰으로 등록된다.. 1번나온 단어는 버린다.
# 특성데이터가 너무많으면 적절하게 조절함.
vectorizer = CountVectorizer(min_df = 2)
vectorizer.fit(corpus)

print("토큰의 개수 :", len(vectorizer.vocabulary_))
print("토큰의 내용 :", (vectorizer.vocabulary_))

# 희소행렬의 결과 - one out 
print("변환 결과(희소행렬) :",vectorizer.transform(['This is the second document']))
# 문자열 분석의 용도 - 평점같은거..
print("변환 결과(밀집행렬) :",vectorizer.transform(['This is the second document']).toarray())

# 다수개의 문서를 입력받아 각 문서를 BOW 인코딩 벡터로 변환하는 예제 
print("변환 결과(다수개의 문자) :",vectorizer.transform(['This is the second document',
          'Hello Python', 
          'This is the first document',
          'This is the second document']).toarray())

# 문자 처리할때는 일단 말뭉치를 가지고 인코딩해서 배열을 만듬.
# 각 문서(말뭉치를 구성하는 각 샘플데이터)는 각 토큰으로 나누고 숫자를 먹이고 행렬로 표현
# 글자수는 다다르지만 행렬로 표현하게 되면 특성데이터의 개수가 같아지는것과 같다.  