# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:37:21 2021

@author: MJH
#refer: https://keras.io/examples/nlp/semantic_similarity_with_bert/
"""
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    GlobalAveragePooling1D, 
    GlobalMaxPooling1D,
    concatenate,
    Dropout,
    Dense
    )
from tensorflow.keras.optimizers import Adam
from transformers import BertConfig, TFBertModel



class Similarity:
    
    def __init__(self, config_path, model_path, max_len):
        self.max_len = max_len
        self.config = config_path
        self.model_path = model_path    
    
        
    def build_model(self):
    
        input_ids = Input(
            shape = (self.max_len, ), dtype = tf.int32, name = 'input_ids'
        )
        attention_mask = tf.keras.layers.Input(
            shape = (self.max_len, ), dtype = tf.int32, name = 'attention_masks'
        )
        token_type_ids = tf.keras.layers.Input(
            shape = (self.max_len, ), dtype = tf.int32, name = 'segment_ids'
        )
        
        # model load
        config = BertConfig.from_pretrained(self.config, output_hidden_states = True)
        bert_model = TFBertModel.from_pretrained(self.model_path, from_pt = True, config = config)
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False
    
        outputs = bert_model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        sequence_output, _ = outputs.last_hidden_state, outputs.pooler_output
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = Bidirectional(
            LSTM(units = 64, return_sequences = True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = GlobalAveragePooling1D()(bi_lstm)
        max_pool = GlobalMaxPooling1D()(bi_lstm)
        concat = concatenate([avg_pool, max_pool])
        dropout = Dropout(rate = 0.3)(concat)
        output = Dense(units = 3, activation = 'softmax')(dropout)
        model = Model(
            inputs = [input_ids, attention_mask, token_type_ids], 
            outputs = output
        )
    
        model.compile(
            optimizer = Adam(),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'],
        )

        return model