# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:16:42 2021

@author: MJH
"""
from auxiliary import semantic_similarity
from model import semantic_similarity_model



if __name__ == '__main__':
    
    max_len = 128
    
    model = semantic_similarity_model(max_len).build_model('model')
    model.load_weights('model/semantic_similarity')
    
    get_similarity = semantic_similarity(model)
    
    sentence1 = ''
    sentence2 = ''
    
    get_similarity(sentence1, sentence2)