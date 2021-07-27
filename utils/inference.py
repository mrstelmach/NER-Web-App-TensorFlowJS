# -*- coding: utf-8 -*-

"""Running model in inference mode."""


from dataclasses import dataclass

import numpy as np
from tensorflow.keras import Model as KerasModel

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
