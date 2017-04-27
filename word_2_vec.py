import word2vec
import numpy

def get_model():
    return word2vec.load('data/text8.bin')

def get_cosine_sim(phrase1, phrase2, model):
    if phrase1 not in model or phrase2 not in model:
        return 0
    cos_sim = numpy.dot(model[phrase1], model[phrase2])/(numpy.linalg.norm(model[phrase1])* numpy.linalg.norm(model[phrase2]))
    return cos_sim
