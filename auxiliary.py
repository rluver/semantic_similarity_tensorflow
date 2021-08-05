# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:54:00 2021

@author: MJH
#refer: https://keras.io/examples/nlp/semantic_similarity_with_bert/
"""
from tokenization import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences



def categorizer(label):
    
    if label == 'entailment':
        return 2
    elif label == 'neutral':
        return 1
    else:
        return 0
    
    
    
class semantic_similarity:
    
    def __init__(self, model):
        self.model = model
        self.labels = ['contradiction', 'neutral', 'entailment']
        
        
    def __call__(self, sentence1, sentence2):
        
        return self.get_similarity(sentence1, sentence2)
    
        
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
        proba = f"{float(proba[idx]) * 100: .2f}%"
        pred = self.labels[idx]
        
        return pred, proba




class BertSemanticDataGenerator(Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size,
        shuffle = True,
        include_targets = True,
        max_len = 128
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets        
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = FullTokenizer('vocab.korean.rawtext.list')
        self.max_len = max_len
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size
    
    

    def get_batch_bert_input_data(self, sentence_pairs):
        
    
        sentence_pairs = list(map(lambda x: ' '.join(['[CLS]', x[0], '[SEP]', x[1], '[SEP]']), sentence_pairs))
    
        input_ids = map(lambda x: self.tokenizer.wordpiece_tokenizer.tokenize(x), sentence_pairs)
        input_ids = list(map(lambda x: self.tokenizer.convert_tokens_to_ids(x), input_ids))
                
        mask_array = list(map(lambda x: [1] * len(x), input_ids))
        input_mask_array = pad_sequences(mask_array, maxlen = self.max_len, padding = 'post')
        
        segment_index_lists = list(map(lambda x: np.where(x == tf.constant(3))[0], input_ids))
        input_segment_array = list(map(lambda x: ( [0] * (x[0] + 1) ) + [1] * ( x[1] - x[0] ), segment_index_lists))
        input_segment_array = pad_sequences(input_segment_array, maxlen = self.max_len, padding = 'post')
        
        input_id_array = pad_sequences(input_ids, maxlen = self.max_len, padding = 'post', dtype = 'int32')
        
        return [input_id_array, input_mask_array, input_segment_array]
    
        

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype = 'int32')
            return self.get_batch_bert_input_data(sentence_pairs), labels
        else:
            return self.get_batch_bert_input_data(sentence_pairs)


    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
