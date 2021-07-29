# -*- coding: utf-8 -*-

"""Running model in inference mode."""


import argparse
import os
from dataclasses import dataclass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model as KerasModel

from model_tf.model_tuner import tf_set_memory_growth
from utils.text import CustomTokenizer, TextPreprocessor


@dataclass
class getPredictedNER:
    """End to end NER prediction for given text."""
    preprocessor: TextPreprocessor
    tokenizer: CustomTokenizer
    model: KerasModel
    labels: list
    
    def __call__(self, input_text):
        input_text = self.preprocessor(input_text)
        input_X = self.tokenizer.transform(input_text)
        input_mask = np.squeeze(input_X[-1]).astype(bool)
        input_tokens = self.tokenizer.get_original_tokens(input_text)
        pred_ids = np.argmax(self.model.predict(input_X), axis=-1)
        pred_ids = np.squeeze(pred_ids)[input_mask]
        pred_cls = [self.labels[idx] for idx in pred_ids]
        ner_output = [(tkn, ner) for tkn, ner 
                      in zip(input_tokens, pred_cls)]
        
        return ner_output


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Process txt file with NER model.')
    parser.add_argument('file', help='txt file for model input')
    args = parser.parse_args()
    with open (args.file) as sample_file:
        sample = sample_file.read()
    
    tf_set_memory_growth()
    
    text_prc = joblib.load('inference/text_preprocessor.joblib')
    tokenizer = joblib.load('inference/tokenizer.joblib')
    model = tf.keras.models.load_model('inference/h5_model/model.h5')
    label_encoder = joblib.load('inference/label_encoder.joblib')

    get_ner = getPredictedNER(text_prc, tokenizer, model, label_encoder.classes_.tolist())
    print(get_ner(sample))
    