# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:34:38 2021

@author: MJH
@refer: https://keras.io/examples/nlp/semantic_similarity_with_bert/
"""
from tokenization import *
from model import semantic_similarity_model
from auxiliary import categorizer, BertSemanticDataGenerator

import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam




def main(train_dataset_path, valid_dataset_path, max_len, batch_size_feature, epochs_feature, batch_size_fine_tuning, epochs_fine_tuning, save_path):
    
    train_dataset = pd.read_csv(train_dataset_path, sep = '\t', error_bad_lines = False).dropna().reset_index(drop = True)    
    train_dataset.gold_label = train_dataset.gold_label.apply(lambda x: categorizer(x))
    y_train = to_categorical(train_dataset.gold_label, num_classes = 3)

    
    valid_dataset = pd.read_csv(valid_dataset_path, sep = '\t', error_bad_lines = False).dropna().reset_index(drop = True)
    valid_dataset.gold_label = valid_dataset.gold_label.apply(lambda x: categorizer(x))
    y_valid = to_categorical(valid_dataset.gold_label, num_classes = 3)

    
    
    # feature extraction
    train_data_feature_extraction = BertSemanticDataGenerator(
        train_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_train, 
        batch_size = batch_size_feature, 
        max_len = max_len,
        shuffle = True
        )
    
    valid_data_feature_extraction = BertSemanticDataGenerator(
        valid_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_valid, 
        batch_size = batch_size_feature, 
        max_len = max_len,
        shuffle = False
        )
        
    
    model = semantic_similarity_model(max_len).build_model(base_model_path = 'model')
    model.summary()
            
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        patience = 3
        )
    tensorboard = TensorBoard(
        log_dir = 'log'
        )
    
    history_feature_extraction = model.fit(
        train_data_feature_extraction,
        validation_data = valid_data_feature_extraction,
        epochs = epochs_feature,
        use_multiprocessing = True,
        workers = -1,
        callbacks = [early_stopping, tensorboard]
        )
    
    
    # fine-tuning
    train_data_fine_tuning = BertSemanticDataGenerator(
        train_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_train, 
        batch_size = batch_size_fine_tuning, 
        max_len = max_len,
        shuffle = True
        )
    
    valid_data_fine_tuning = BertSemanticDataGenerator(
        valid_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_valid, 
        batch_size = batch_size_fine_tuning, 
        max_len = max_len,
        shuffle = False
        )
    
    
    
    model.trainable = True
    model.compile(
        optimizer = Adam(1e-5),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
        )
    model.summary()

    history_fine_tuning = model.fit(        
        train_data_fine_tuning,
        validation_data = valid_data_fine_tuning,
        epochs = epochs_fine_tuning,
        use_multiprocessing = True,
        workers = -1,
        callbacks = [early_stopping, tensorboard]
        )
    
    
    model.save_weights(save_path)


if __name__ == '__main__':
    
    labels = ['contradiction', 'neutral', 'entailment']
    
    train_dataset_path = r'dataset\multinli.train.ko.tsv.txt'
    valid_dataset_path = r'dataset\xnli.test.ko.tsv.txt'
    
    max_len = 512
    batch_size_feature = 512
    epochs_feature = 20
    
    batch_size_fine_tuning = 24
    epochs_fine_tuning = 20
    
    save_path = 'news_model'
    
    main(train_dataset_path, valid_dataset_path, max_len, batch_size_feature, epochs_feature, batch_size_fine_tuning, epochs_fine_tuning, save_path)