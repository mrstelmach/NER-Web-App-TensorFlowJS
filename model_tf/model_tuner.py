# -*- coding: utf-8 -*-

"""Hyperparameter tuning."""


import json

import joblib
import keras_tuner as kt
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.layers import (Bidirectional, Concatenate, Conv1D,
                                     Dense, Dropout, Embedding, GlobalMaxPool1D, 
                                     Input, LSTM, TimeDistributed)

from utils.text import get_glove_embedding_matrix


def tf_set_memory_growth():
    """Set memory growth option for GPU device."""
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu_devices) > 0:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def build_model(hp):
    """Build Keras model with hyperparameters."""
    
    # hyperparameters
    dns_units = hp.Int('dns_units', min_value=160, max_value=256, step=4)
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=160, step=4)
    ch_emb_dim = hp.Int('ch_emb_dim', min_value=8, max_value=32, step=2)
    conv_filters = hp.Int('conv_filters', min_value=16, max_value=48, step=2)
    conv_kernel = hp.Choice('conv_kernel', values=[2, 3, 4, 5])
    drp = hp.Float('drp', min_value=0.2, max_value=0.5, step=0.025)
    lr = hp.Choice('lr', values=[5e-4, 7.5e-4, 1e-3])

    # mask input on word sequence level
    mask_inp = Input(shape=(conf['MAX_SEQ_LEN'],), name="mask_input")
    
    # word level input
    word_inp = Input(shape=(conf['MAX_SEQ_LEN'],), name='word_input')
    word_emb = Embedding(input_dim=conf['WORD_VOCAB_SIZE'], 
                         output_dim=EMBEDDING_MATRIX.shape[1], mask_zero=True, 
                         input_length=conf['MAX_SEQ_LEN'],
                         weights=[EMBEDDING_MATRIX], trainable=False, 
                         name='word_embedding')(word_inp)
    
    # character level input
    char_inp = Input(shape=(conf['MAX_SEQ_LEN'], conf['MAX_WRD_LEN']), 
                     name="character_input")
    char_emb = TimeDistributed(
        Embedding(input_dim=conf['CHAR_VOCAB_SIZE'], output_dim=ch_emb_dim, 
                  input_length=conf['MAX_WRD_LEN']),
        name="character_embedding")(char_inp)
    conv1d = TimeDistributed(
        Conv1D(filters=conv_filters, kernel_size=conv_kernel, 
               strides=1, padding='same', activation='tanh'), 
        name="1d_character_convolution")(char_emb)
    char_features = TimeDistributed(
        GlobalMaxPool1D(), 
        name="global_max_pooling")(conv1d)
    
    # concatenation and output
    conc = Concatenate(name='concatenation')([word_emb, char_features])
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True),
                         name='bi_lstm')(conc)
    dns = TimeDistributed(
        Dense(dns_units, activation='relu'), 
        name='dense')(lstm)
    dns_drop = Dropout(drp, name='dense_dropout')(dns)
    out = TimeDistributed(
        Dense(conf['NUM_CLASSES'], activation='softmax'), 
        name='output')(dns_drop, mask=mask_inp)
    
    model = tf.keras.Model([word_inp, char_inp, mask_inp], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  weighted_metrics=['sparse_categorical_crossentropy',
                                    'sparse_categorical_accuracy'])
    
    return model


if __name__ == '__main__':
    
    # load config file, development data and pretrained embeddings 
    # for building model and hyperparameter tuning with Bayesian Optimization; 
    # save best model in h5 format and convert to tfjs for web app deployment;
    # evaluate on train, validation and test datasets;
    
    tf_set_memory_growth()
    
    with open('config.json') as config_file:
        conf = json.load(config_file)
        
    data = joblib.load('data/data.joblib')
    X_train, y_train, y_train_mask = data['train']
    X_valid, y_valid, y_valid_mask = data['valid']
    X_test, y_test, y_test_mask = data['test']
    
    word_index = (joblib.load('inference/tokenizer.joblib')
                  .word_tokenizer.word_index)
    EMBEDDING_MATRIX = get_glove_embedding_matrix(
        'data/glove.6B.100d.txt', word_index
    )
    
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_crossentropy', min_delta=0, 
        patience=2, verbose=0, mode='min', baseline=None, 
        restore_best_weights=True
        )

    tuner = kt.BayesianOptimization(
        hypermodel=build_model, objective='val_sparse_categorical_crossentropy',
        max_trials=100, num_initial_points=10, directory='model_tf',
        project_name='hptuning'
        )
    
    fit_args = {'x': X_train, 'y': y_train, 'sample_weight': y_train_mask, 
                'validation_data': (X_valid, y_valid, y_valid_mask), 
                'epochs': 50, 'batch_size': 64, 'shuffle': True,
                'verbose': 2, 'callbacks': [es_callback]}

    tuner.search(**fit_args)
    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hparams)
    print(model.summary())
    model.fit(**fit_args)
    
    model.save('inference/h5_model/model.h5')
    tfjs.converters.save_keras_model(model, 'web-app/tfjs_model')
    
    print('\n')
    print('Categorical crossentropy and Accuracy for training data: {}.'.format(
        model.evaluate(x=X_train, y=y_train, sample_weight=y_train_mask, verbose=0)[1:]))
    print('Categorical crossentropy and Accuracy for validation data: {}.'.format(
        model.evaluate(x=X_valid, y=y_valid, sample_weight=y_valid_mask, verbose=0)[1:]))
    print('Categorical crossentropy and Accuracy for test data: {}.'.format(
        model.evaluate(x=X_test, y=y_test, sample_weight=y_test_mask, verbose=0)[1:]))
