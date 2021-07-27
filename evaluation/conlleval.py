# -*- coding: utf-8 -*-

"""Perl evaluation script in Python."""


import os

import numpy as np

from utils.encoders import indices_to_labels
from utils.text import create_eval_file


def evaluate(sentences, labels, label_encoder, X, y_mask, model,
             directory='evaluation', pred_file='pred.txt',
             eval_file='eval.txt', script_file='conlleval.pl'):
    """
    Create evaluation file with script_file Perl script and print 
    the evaluation result.
    """
    
    pred_path = os.path.join(directory, pred_file)
    eval_path = os.path.join(directory, eval_file)
    script_path = os.path.join(directory, script_file)
    
    pred_ids = model.predict(X)
    pred_ids = np.argmax(pred_ids, axis=-1)
    pred_labels = indices_to_labels(
        label_encoder, pred_ids, y_mask.astype(bool)
    )
    create_eval_file(pred_path, sentences, labels, pred_labels)
    eval_command = '{} < {} > {}'.format(
        script_path, pred_path, eval_path
    )
    os.system(eval_command)
    
    with open(eval_path) as f:
        evaluation = f.read()
    
    print(evaluation)
