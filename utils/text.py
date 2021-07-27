# -*- coding: utf-8 -*-

"""Set of utility functions for working with text."""


import json
import re

import numpy as np
import tensorflow as tf


def read_txt_file(path):
    """Read txt file with tagged lines."""
    sentences, labels = [], []
    with open(path) as infile:
        words, tags = [], []
        for line in infile:
            if line != '\n':
                if line == '-DOCSTART- -X- -X- O\n':
                    continue
                else:
                    line = line.rstrip().split()
                    words.append(line[0])
                    tags.append(line[-1])
            elif bool(words):
                sentences.append(words)
                labels.append(tags)
                words, tags = [], []
    return sentences, labels


def write_js_from_dict(path, dictionary, const_name='constname'):
    """Write python dictionary to specified js file."""
    with open(path, 'w') as f:
        f.write('const {} = {}'.format(
            const_name, json.dumps(dictionary, indent=4)))


def create_eval_file(path, tokenized_sentences, labels_per_sentece, 
                     predictions_per_sentece):
    """
    Create file for evaluation. Sentences, labels and predictions should be
    all nested lists with each sublist representing single sentence:
        
        sentences: [['EU', 'rejects', 'German', 'call', '.']]
        labels: [['B-ORG', 'O', 'B-MISC', 'O', 'O']]
        predictions: [['B-ORG', 'O', 'B-PER', 'O', 'O']]
        
    Output created is of txt format and vertical form:
        
        EU B-ORG B-ORG
        rejects O O
        German B-MISC B-PER
        call O O
        . O O
    
    """
    
    chunks = zip(tokenized_sentences, labels_per_sentece,
                 predictions_per_sentece)
    
    with open(path, 'w') as write_file:
        for tokens, labels, predictions in chunks:
            for token, label, prediction in zip(tokens, labels, predictions):
                write_file.write('{}\n'.format(
                    ' '.join([token, label, prediction])))
            write_file.write('\n')


def get_glove_embedding_matrix(glove_file_path, word_index_dict, 
                               dtype='float32'):
    """
    Import glove embeddings from file to python dictionary and create
    embedding matrix according to word indices in word_index_dict.
    """
    
    glove_embeddings = {}
    with open (glove_file_path, 'r', encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=dtype)
            glove_embeddings[word] = vector
    
    embedding_matrix = np.random.random(
        (len(word_index_dict), len(vector))
        ).astype(dtype)
    
    for wrd, idx in word_index_dict.items():
        if wrd in glove_embeddings.keys():
            embedding_matrix[idx] = glove_embeddings[wrd]
    
    return embedding_matrix
            

class TextPreprocessor:
    """Class for preprocessing text input."""
    
    def __init__(self, separate_apostrophes=True, separate_punctuation=True,
                 punctuation='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'):
        self.separate_apostrophes = separate_apostrophes
        self.separate_punctuation = separate_punctuation
        self.punctuation = punctuation
    
    def _get_apostrophes_separated(self, text):
        """
        Insert space before apostrophes indicating possession or 
        used in contractions, e.g. Peter's -> Peter 's or I'm -> I 'm.
        """
        eos = (self.punctuation + " ")
        text = re.sub(r"(?i)([a-z])('|'s|'ll|'m|'re|'ve|'d|n't)([{}]|$)"
                      .format(eos), r"\1 \2\3", text)
        
        return text
    
    def _get_punctuation_separated(self, text):
        """Separate punctuation from other text."""
        text = re.sub(r"(?i)(\S)([{}])".format(self.punctuation), 
                      r"\1 \2", text)
        text = re.sub(r"(?i)([{}])(\S)".format(self.punctuation), 
                      r"\1 \2", text)
        
        return text
    
    def __call__(self, text):
        """Apply preprocessing steps."""
        
        return_string = False
        msg = 'String or list of strings required as input.'
        
        if isinstance(text, list):
            if not all([isinstance(t, str) for t in text]):
                raise TypeError(msg)
        elif isinstance(text, str):
            return_string = True
            text = [text]
        else:
            raise TypeError(msg)
        
        if self.separate_apostrophes:
            text = [self._get_apostrophes_separated(t) for t in text]
        
        if self.separate_punctuation:
            text = [self._get_punctuation_separated(t) for t in text]
        
        if return_string:
            return text[0]
            
        return text


class CustomTokenizer:
    """
    Tokenize texts on word level and optionally on character level.
    Character level tokenizer proceeds on cased text.
    """
    
    def __init__(self, char_level=True, oov_token='[UNK]', 
                 pad_token='[PAD]', filters='', lower=True):
        self.char_level = char_level
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.filters = filters
        self.lower = lower
        self._fitted = False
        
        self.word_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters=filters, lower=lower, char_level=False, 
            oov_token=oov_token)
        
        if self.char_level:
            self.char_tokenizer = tf.keras.preprocessing.text.Tokenizer(
                filters=filters, lower=False, char_level=True, 
                oov_token=oov_token)
    
    def _check_if_fitted(self):
        """Raise error if tokenizer is not fitted."""
        if not self._fitted:
            raise AttributeError('CustomTokenizer not fitted yet.')
    
    def fit(self, texts, max_seq_len=32, max_word_len=16):
        """Fit tokenizers on texts."""
        
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        
        self.word_tokenizer.fit_on_texts(texts)
        self.word_tokenizer.word_index[self.pad_token] = 0
        
        if self.char_level:
            self.char_tokenizer.fit_on_texts(texts)
            self.char_tokenizer.word_index[self.pad_token] = 0
            
        self._fitted = True

        return self
    
    def get_original_tokens(self, text):
        """Retrieve original word level tokens (e.g., before lowering)."""
        
        if not isinstance(text, str):
            raise TypeError('String type input required.')
        
        self._check_if_fitted()
        
        if bool(self.filters):
            text = re.sub(f'[{self.filters}]', ' ', text)
        
        return text.split()[:self.max_seq_len]
    
    def transform(self, texts, dtype='int32'):
        """
        Transform texts with fitted tokenizers and pad.
        List of texts, list of lists of texts or standalone text 
        is acceptable as input.
        """
        
        self._check_if_fitted()
        
        if isinstance(texts, str):
            texts = [texts]
        
        tokenized_word_ids = self.word_tokenizer.texts_to_sequences(texts)
        tokenized_word_ids = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_word_ids, maxlen=self.max_seq_len, padding='post', 
            truncating='post').astype(dtype)
        input_mask = (tokenized_word_ids != 0).astype(dtype)
        
        if self.char_level:
            if not all(isinstance(l, list) for l in texts):
                if bool(self.filters):
                    texts = [re.sub(f'[{self.filters}]', ' ', text) 
                             for text in texts]
                texts = [text.split() for text in texts]
            
            empty_padded_texts = \
                tf.keras.preprocessing.sequence.pad_sequences(
                texts, maxlen=self.max_seq_len, padding='post', 
                truncating='post', value='', dtype=object
                )
            
            tokenized_char_ids = [
                self.char_tokenizer.texts_to_sequences(text)
                for text in empty_padded_texts
                ]
            
            tokenized_char_ids = [
                tf.keras.preprocessing.sequence.pad_sequences(
                tokens, maxlen=self.max_word_len, padding='post',
                truncating='post') 
                for tokens in tokenized_char_ids
                ]
            tokenized_char_ids = np.array(tokenized_char_ids).astype(dtype)
        
            return tokenized_word_ids, tokenized_char_ids, input_mask
        
        return tokenized_word_ids, input_mask
