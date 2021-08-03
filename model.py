# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:34:38 2021

@author: MJH
#refer: https://keras.io/examples/nlp/semantic_similarity_with_bert/
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from transformers import BertConfig, TFBertModel




class semantic_similarity_model:
    
    def __init__(self, max_length):
        self.max_length = max_length 
        
        
    def _get_model(self, base_model_path):
        
        input_ids = Input(
            shape = (self.max_length, ), dtype = tf.int32, name = 'input_ids'
        )
        token_type_ids = tf.keras.layers.Input(
            shape = (self.max_length, ), dtype = tf.int32, name = 'segment_ids'
        )
        attention_masks = tf.keras.layers.Input(
            shape = (self.max_length, ), dtype = tf.int32, name = 'attention_masks'
        )


        # model load
        config = BertConfig.from_pretrained(base_model_path, output_hidden_states = True)
        bert_model = bert_model = TFBertModel.from_pretrained(base_model_path, from_pt = True, config = config)
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False

        outputs = bert_model([input_ids, token_type_ids, attention_masks])
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
            inputs = [input_ids, token_type_ids, attention_masks], 
            outputs = output
        )

        model.compile(
            optimizer = Adam(),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'],
        )

        return model
        
        
    
    def build_model(self, base_model_path, gpus: int = -1):
    
        if gpus >= 2:            
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                return self._get_model(base_model_path)
    
        else:
            return self._get_model(base_model_path)