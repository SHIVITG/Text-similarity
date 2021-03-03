import sys
import logging
from operator import itemgetter
from keras.models import load_model
from semantic_nn.inputHandler import word_embed_meta_data, create_test_data
from semantic_nn.config import siamese_config

'''Global Parameters'''
# path to pretrained model
model = load_model('text-sim-model/semantic_nn/data/checkpoints/1608380268/lstm_150_150_0.17_0.25.h5')

class TextSim():

    def __init__(self,sen1,sen2):
        self.sen1 = sen1
        self.sen2 = sen2
    
    def fetch_test_data(self):
        test_sentence_pairs = [(self.sen1,self.sen2)]
        tokenizer, embedding_matrix = word_embed_meta_data(self.sen1 + self.sen2,siamese_config['EMBEDDING_DIM'])
        embedding_meta_data = {
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        }
        test_data_x1,test_data_x2,leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
        return test_sentence_pairs,test_data_x1,test_data_x2,leaks_test

    def predict_score(self):
        test_sentence_pairs,test_data_x1,test_data_x2,leaks_test = TextSim.fetch_test_data(self)
        preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
        results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
        results.sort(key=itemgetter(2), reverse=True)
        return results

if __name__ == "__main__":
    sentence1,sentence2 = 'What can make python easy to learn?','How can you make python easy to learn?'
    textsim = TextSim(sentence1,sentence2)
    print(textsim.predict_score())