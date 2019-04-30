# -*- coding: utf-8 -*-

# TfidfVectorizer 클래스
# CountVectorizer와 비슷하지만 TF-IDF 방식으로 
# 단어의 가중치를 조정한 BOW 벡터를 생성
# TF-IDF(Term Frequency – Inverse Document Frequency)
# TF : 특정한 단어의 빈도수
# IDF : 특정한 단어가 들어 있는 문서의 수에 반비례하는 수

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'This is the first document.',
    'This is the second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',    
]

vect = TfidfVectorizer().fit(corpus)

print('토큰 개수 : ', len(vect.vocabulary_))
print('토큰 내용 : \n', vect.vocabulary_)
print('단어 벡터 : \n', vect.transform(corpus).toarray())

print("=" * 17)

vect = TfidfVectorizer(stop_words="english").fit(corpus)

print('토큰 개수 : ', len(vect.vocabulary_))
print('토큰 내용 : \n', vect.vocabulary_)
print('단어 벡터 : \n', vect.transform(corpus).toarray())














