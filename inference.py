# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:16:42 2021

@author: MJH
"""
from auxiliary import semantic_similarity
from model import semantic_similarity_model



if __name__ == '__main__':
    
    model = semantic_similarity_model(128).build_model('model')
    model.load_weights('model/semantic_similarity')
    
    get_similarity = semantic_similarity(model)
    
    get_similarity('안녕하세요 여러분', '반갑습니다')