
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Hello Python', 
          'This is the first document',
          'This is the second document',
          'And thr third document',
          'is this the first document?',
          'The last document']

# stop_words 생성자 매개변수 
# stop_words 문자열, 리스트 또는 None
# 토큰 생성 시 제외하고 하느 ㄴ단어를 리스트로 전달.
vectorizer = CountVectorizer(stop_words=['is','the'])


# 영어중에 의미없는 단어를 지우고 싶을 때 (부용어처리 리스트가 내부적으로 있음)
# stop_words='english' 로 지정하면 불용으로 인식 된 단어를 제외하고 토큰을 생성함.
vectorizer = CountVectorizer(stop_words='english') 
vectorizer.fit(corpus)

print("토큰의 개수 :", len(vectorizer.vocabulary_))
print("토큰의 내용 :", (vectorizer.vocabulary_))

print("변환 결과(희소행렬) :",vectorizer.transform(['This is the second document']))
print("변환 결과(밀집행렬) :",vectorizer.transform(['This is the second document']).toarray())

print("변환 결과(다수개의 문자) :",vectorizer.transform(['This is the second document',
          'Hello Python', 
          'This is the first document',
          'This is the second document']).toarray())
