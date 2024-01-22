import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pickle


def tagged_data(data1,data2):
    docs = []
    docs.append(TaggedDocument(words=word_tokenize(data1.lower()), tags=['0']))
    docs.append(TaggedDocument(words=word_tokenize(data2.lower()), tags=['1']))
    return docs


def model_training(data):
    model = Doc2Vec(vector_size=30, min_count=1, epochs=80)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=80)
    model.save("d2v.model")

def similarity_score(tag_number):
    
    model1 =Doc2Vec.load("d2v.model") 
    similar_doc = model1.docvecs.most_similar(str(tag_number))
    return similar_doc


