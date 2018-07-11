# from gensim.models.word2vec import Word2Vec, wmdistance
from gensim.models.word2vec import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
model.save('./data/MyModel')
