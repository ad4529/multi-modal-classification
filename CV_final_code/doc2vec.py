from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from pathlib import Path


def train_vec_rep(data):

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    max_epochs = 100
    vec_size = 50176
    alpha = 0.025

    model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)

        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    print("Generated Model")
    return model

def make_vector(cat_features):
    temp_feats_one = []

    for i in range(50):
        temp_feats_one.append(cat_features['years'][i] + ' ' + cat_features['genre'][i] + ' ' + cat_features['nationality'][i])

    file_path = Path('/home/ad0915/Desktop/CV_final/bio_feats.model')
    if not file_path.is_file():
        model = train_vec_rep(temp_feats_one)
        model.save('cat_feats.model')
        model_bio = train_vec_rep(cat_features['bio'])
        model_bio.save('bio_feats.model')
    else:
        model = Doc2Vec.load('cat_feats.model')
        print('Done loading basic features model')
        model_bio = Doc2Vec.load('bio_feats.model')
        print('Done loading bio features model')

    d = {}

    for i in range(50):
        temp_vec = np.absolute(model.docvecs[str(i)])
        temp_bio_vec = np.absolute(model_bio.docvecs[str(i)])
        d[cat_features['name'][i]] = [temp_vec.reshape((224,224,1))/np.amax(temp_vec), temp_bio_vec.reshape((224,224,1))/np.amax(temp_bio_vec)]
        print('Processed {} artist info'.format(i))
    return d
