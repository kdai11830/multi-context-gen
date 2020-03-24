import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import os
import platform
import xml.etree.ElementTree as ET
import html
from itertools import chain
import shutil
import re
import random


plf = platform.system()
split_key = ''

if 'Windows' in plf:

    split_key = '\\'
else:
    split_key = '/'


def load_data(rootdir, n_files=0):
    # print("start loading data")
    # print(rootdir)
    data = []
    i = -1

    for subdir, dirs, files in os.walk(rootdir):
        # print(subdir)

        if n_files != 0 and i == n_files:
            return data
        # print(subdir)
        personality = ''
        context_1 = ''
        context_2 = ''
        if subdir.split(split_key)[-1] != 'Talk':
            # print("asdkjfhaksdfhkahsdkjfhakjsdh")
            filename = subdir.split(split_key)[-1].split('_')
            # print(filename)
            del filename[-1]
            personality = filename[0]
            context_1 = filename[1]
            context_2 = '_'.join(filename[2:])

        for file in files:
            if 'xml' in file and '1' not in file and '2' not in file: # only want 00000000.kup because the rest aren't english
                with open(subdir + split_key + file, 'r', encoding='utf-8') as tree:
                    raw = html.unescape(tree.readlines())

                    # prune data for usable information
                    cur = ''
                    started = False
                    for line in raw:
                        # print(line[:40])
                        if '<original>' in line and not started:
                            # print(line[:40])
                            cur = personality + ' ' + context_1 + ' ' + context_2
                            started = True
                            cur += line
                        elif '</original>' in line and started:
                            started = False
                            cur += line
                            cur = cur.split('</original>', 1)[0]
                            cur = cur.replace('<original>', '')
                            # cur = cur.replace('</original>', '')
                            # print(cur, '\n')
                            data.append(cur.lower())
                            cur = ''
                        elif started:
                            cur += line

        i += 1

    return data

def clean_data(data):
    new_data = []
    regex = '&#x[0-9A-Za-z]{1,2};'
    for d in data:
        temp = d
        temp = re.sub('<original>', '', temp)
        temp = re.sub('</original>', '', temp)
        temp = re.sub('<edited>', ' ', temp)
        temp = re.sub('</edited>', ' ', temp)
        temp = re.sub(regex, ' ', temp)
        temp.replace('\\t', ' ')
        temp.replace('\\n', ' ')
        temp.replace('\'', '')
        temp = re.sub('[^0-9a-zA-Z ]+', ' ', temp)
        temp = re.sub('[ ]+', ' ', temp)
        #print(temp)
        new_data.append(temp)
    return new_data

def load_data_edited(rootdir):
    data = []
    directory = os.fsencode(rootdir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open('data' + split_key + '3-classes-replaced-tags' + split_key + filename, 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            data = data + content

    return data

def tokenize_data(data, lower=True):
    tokenized = [word_tokenize(s.lower()) if lower else word_tokenize(s) for s in data]
    return tokenized
    

################################## DATA ENCODING ##################################

# Defining method to encode one hot labels
def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot
    
# Defining method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

############################## CREATE EMBEDDINGS ################################

def get_pretrained_weights(model_path=None, data=None, lower=True):
    if data is None:
        print("WRONG INPUTS")
        return

    # load model if exists
    if model_path is not None:
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)

    # else train new w2v model

    # tokenize into words
    word_tokens = tokenize_data(data, lower)
    if model_path is None:
        model = Word2Vec(word_tokens, min_count=1) # for now use default params, later cross validate(?)
        model.save('models' + split_key + 'trained' + split_key + 'w2v_embeddings.model')
        # return model.wv, word_tokens
    return model, word_tokens


def create_embedding(weights, non_trainable=False):
    n_embeddings, embedding_dim = weights.shape
    # n_embeddings = len(weights.vectors)
    # embedding_dim
    print("embedding dim: ",embedding_dim)
    # n_embeddings = len(weights.vocab)
    embedding = nn.Embedding(n_embeddings, embedding_dim)
    embedding.load_state_dict({'weight': weights})
    if non_trainable:
        embedding.weight.requires_grad = False

    return embedding, embedding_dim