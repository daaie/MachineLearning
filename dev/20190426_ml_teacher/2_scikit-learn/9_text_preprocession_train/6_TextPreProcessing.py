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

vectorizer = TfidfVectorizer().fit(corpus)

print("토큰 개수 : ", len(vectorizer.vocabulary_))
print("토큰 내용 : ", vectorizer.vocabulary_)
print("변환 결과 : \n", vectorizer.transform(corpus).toarray())

print("=" * 30)

vectorizer = TfidfVectorizer(min_df=3).fit(corpus)

print("토큰 개수 : ", len(vectorizer.vocabulary_))
print("토큰 내용 : ", vectorizer.vocabulary_)

print("=" * 30)

vectorizer = TfidfVectorizer(stop_words='english').fit(corpus)

print("토큰 개수 : ", len(vectorizer.vocabulary_))
print("토큰 내용 : ", vectorizer.vocabulary_)













