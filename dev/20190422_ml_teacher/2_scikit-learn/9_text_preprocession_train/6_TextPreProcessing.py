# -*- coding: utf-8 -*-

# TfidfVectorizer 클래스
# CountVectorizer와 비슷하지만 TF-IDF 방식으로 
# 단어의 가중치를 조정한 BOW 벡터를 생성
# TF-IDF(Term Frequency – Inverse Document Frequency)
# TF : 특정한 단어의 빈도수
# IDF : 특정한 단어가 들어 있는 문서의 수에 반비례하는 수
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['Hello Python', 
          'This is the first document',
          'This is the second document',
          'And the third document',
          'is this the first document?',
          'The last document']

# 매개변수가 CountVectorizer와 비슷. 카운트벡터라이저를 상속받ㄷ아 만듬.
vectorizer = TfidfVectorizer().fit(corpus)

print("토큰 개수",len(vectorizer.vocabulary_))
print("토큰 내용",(vectorizer.vocabulary_))

# 어느정도의 빈도로 단어가 나오는지 출력.
# 너무 많이나오면 가중치가 떨어짐
# 하나만 나오는건 가중치가 높아짐.
# 하나만 나오는 놈이 분류하기 편하니까 가중치가 높음.

print("변환 결과", vectorizer.transform(corpus).toarray())


vectorizer = TfidfVectorizer(min_df=3).fit(corpus)

print("토큰 개수",len(vectorizer.vocabulary_))
print("토큰 내용",(vectorizer.vocabulary_))


vectorizer = TfidfVectorizer(stop_words='english').fit(corpus)

print("토큰 개수",len(vectorizer.vocabulary_))
print("토큰 내용",(vectorizer.vocabulary_))

