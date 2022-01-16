from sentence_transformers import SentenceTransformer

model = SentenceTransformer(r'./sroberta')
negative_vector = model.encode('bad negative')
positive_vector = model.encode('good positive')
