from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1. Токенизация
texts = [text.split() for text in corpus]

# 2. Обучение Word2Vec
w2v = Word2Vec(
    sentences=texts,
    vector_size=300,
    window=5,
    min_count=5,
    workers=4
)

# 3. TF-IDF
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
tfidf_vocab = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# 4. Функция эмбеддинга документа
def embed(text):
    words = text.split()
    vectors = []
    
    for word in words:
        if word in w2v.wv:
            weight = tfidf_vocab.get(word, 1.0)
            vectors.append(w2v.wv[word] * weight)
    
    if len(vectors) == 0:
        return np.zeros(300)
    
    return np.mean(vectors, axis=0)
