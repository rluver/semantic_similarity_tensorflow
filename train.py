# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:45:28 2021

@author: MJH
#refer: https://keras.io/examples/nlp/semantic_similarity_with_bert/
"""
from model import Similarity
from auxiliary import categorizer, BertSemanticDataGenerator

import pandas as pd
import tesnorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard



        
        
def main(train_dataset_path, valid_dataset_path, config_path, model_path, batch_size, max_len, gpus):
    
    train_dataset = pd.read_csv(train_dataset_path, sep = '\t', error_bad_lines = False).dropna().reset_index(drop = True)    
    train_dataset.gold_label = train_dataset.gold_label.apply(lambda x: categorizer(x))
    y_train = to_categorical(train_dataset.gold_label, num_classes = 3)
    train_data = BertSemanticDataGenerator(
        train_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_train, 
        batch_size = batch_size, 
        max_len = max_len,
        shuffle = True
        )
    
    valid_dataset = pd.read_csv(valid_dataset_path, sep = '\t', error_bad_lines = False).dropna().reset_index(drop = True)
    valid_dataset.gold_label = valid_dataset.gold_label.apply(lambda x: categorizer(x))
    y_valid = to_categorical(valid_dataset.gold_label, num_classes = 3)
    valid_data = BertSemanticDataGenerator(
        valid_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_valid, 
        batch_size = batch_size, 
        max_len = max_len,
        shuffle = False
        )
    
   
    EPOCHS = 20
    labels = ['contradiction', 'neutral', 'entailment']
    
    

    if gpus >= 2:
        strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = Similarity(config_path, model_path, max_len).build_model()
    else:
        model = Similarity(config_path, model_path, max_len).build_model()
    
        
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        patience = 3
        )
    
    tensorboard = TensorBoard(
        log_dir = 'log'
        )
    
    # feature extraction
    history = model.fit(
        train_data,
        validation_data = valid_data,
        epochs = EPOCHS,
        use_multiprocessing = True,
        workers = -1,
        callbacks = [early_stopping, tensorboard]
        )
    
    
    # fine-tuning
    model.trainable = True
    model.compile(
        optimizer = Adam(1e-5),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
        )
    
    model.summary()
    
    history = model.fit(        
        train_data,
        validation_data = valid_data,
        epochs = EPOCHS,
        use_multiprocessing = True,
        workers = -1,
        callbacks = [early_stopping, tensorboard]
        )
    
    



if __name__ == '__main__':
    
    
    train_dataset_path = r'dataset\multinli.train.ko.tsv.txt'
    valid_dataset_path = r'dataset\xnli.test.ko.tsv.txt'
    
    config_path = ''
    model_path = ''
    
    max_len = 128
    batch_size = 32
    
    gpus = 2
    
    main(train_dataset_path, valid_dataset_path, config_path, model_path, max_len, batch_size, gpus)