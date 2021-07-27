# -*- coding: utf-8 -*-

"""Convert numerical values to labels (or vice versa)."""


import itertools

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


def indices_to_labels(label_encoder, pred_array, bool_mask_array=None):
    """
    Create labels from indices predicted by a model and potentially apply 
    a boolean mask for rows with padded indices.
    Given label_encoder should provide an inverse_transform method.
    """
    
    if bool_mask_array is None:
        bool_mask_array = np.ones(pred_array.shape).astype(bool)
    
    pred_labels = [
        label_encoder.inverse_transform(pred[mask]).tolist()
        for pred, mask in zip(pred_array, bool_mask_array)
        ]
    
    return pred_labels


class SequentialLabelEncoder:
    """Class for encoding labels provided in sequential form."""
    
    def __init__(self):
        self.le = LabelEncoder()
    
    def fit(self, labels, max_seq_len=None, pad_value=0.0):
        """
        Fit LabelEncoder to flattened list of labels. Labels for input 
        should be provided as a nested list and each sequence of that list 
        might be of different length.
        If max_seq_len is not provided, maximum length of sequence found
        is used to provide consistent length across resulting array during
        transform.
        """
        
        lbs_flattened = list(itertools.chain.from_iterable(labels))
        self.le.fit(lbs_flattened)
        
        self.pad_value = pad_value
        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.max_seq_len = max([len(seq) for seq in labels])
        
        self._fitted = True
        
        return self
        
    def transform(self, labels, return_mask=True, dtype='int32'):
        """
        Transform labels with fitted encoder and pad to max_seq_len.
        """
        
        if not self._fitted:
            raise AttributeError('SequentialLabelEncoder not fitted yet.')
        
        lbs_encoded = [self.le.transform(seq).tolist() for seq in labels]
        
        lbs_encoded = pad_sequences(lbs_encoded, maxlen=self.max_seq_len, 
                                    padding='post', truncating='post', 
                                    value=-1.0).astype(dtype)
        lbs_mask = (lbs_encoded != -1.0).astype(dtype)
        lbs_encoded[~lbs_mask.astype(bool)] = self.pad_value
        
        if return_mask:
            return lbs_encoded, lbs_mask
        
        return lbs_encoded
