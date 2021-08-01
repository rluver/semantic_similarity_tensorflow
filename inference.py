# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:54:58 2021

@author: MJH
#refer: https://keras.io/examples/nlp/semantic_similarity_with_bert/
"""
from model import Similarity
from auxiliary import BertSemanticDataGenerator

import numpy as np




class Semantic_Similarity:
    
    def __init__(self, config_path, model_path, model_weight_path, max_len):
        self.model = Similarity(config_path, model_path, max_len).build_model()
        self.model.load_weights(model_weight_path)
        self.labels = ['contradiction', 'neutral', 'entailment']
    
    
    def get_similarity(self, sentence1, sentence2):
        
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        test_data = BertSemanticDataGenerator(
            sentence_pairs, 
            labels = None, 
            batch_size = 1, 
            shuffle = False, 
            include_targets = False
        )
    
        proba = self.model.predict(test_data)[0]
        idx = np.argmax(proba)
        proba = f"{(proba[idx] * 100): .2f}%"
        pred = self.labels[idx]
        
        return pred, proba




if __name__ == '__main__':
    
    config_path = 'model'
    model_path = 'model'
    model_weight_path = 'model/semantic_similarity'
    max_len = 128
    
    semantic_similarity = Semantic_Similarity(config_path, model_path, model_weight_path, max_len)
    
    sentence1 = '안녕하세요'
    sentence2 = '반갑습니다'
    
    semantic_similarity.get_similarity(sentence1, sentence2)
