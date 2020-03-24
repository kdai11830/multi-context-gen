# Importing libraries
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

from utils import load_data, clean_data, load_data_edited, tokenize_data, one_hot_encode, get_batches


################################## MODELS ##################################

# Declaring the model
class CharLSTM(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001, train_on_gpu=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.tokens = tokens
        self.int2token = dict(enumerate(self.tokens))
        self.token2int = {ch: ii for ii, ch in self.int2token.items()}
        
        #define the LSTM
        self.lstm = nn.LSTM(len(self.tokens), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)

        self.train_on_gpu = train_on_gpu

        # use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print("training on", torch.cuda.device_count(), "GPUS...")
            self.lstm = nn.DataParallel(self.lstm)

        #define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        #define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.tokens))
      
    
    def forward(self, x, hidden):
                
        #get the outputs and the new hidden state from the lstm
        # print('is cuda: ', hidden[1].is_cuda)
        # print(hidden)

        r_output, hidden = self.lstm(x, hidden)
        
        #pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        
        #put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden



########################################################################################################
########################################################################################################




if __name__ == '__main__':

    plf = platform.system()
    rootdir = ''
    split_key = ''

    if 'Windows' in plf:
        # rootdir = 'C:\\Users\\Kevin Dai\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes-replaced-tags'
        rootdir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes-replaced-tags'
        split_key = '\\'
    else:
        rootdir = '/n/home13/kdai/Edited/3-classes-replaced-tags'
        split_key = '/'

    # Check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU!')
    else: 
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    # data = load_data(rootdir, n_files=0)
    # data = clean_data(data)
    data = load_data_edited(rootdir)
    random.shuffle(data)
    print('data length: ', len(data))
    text = '\n'.join(data)
    print('text length: ', len(text))

    print('--------EXAMPLE--------\n',text[:1000],'\n------------------------\n')

    # print(text)
    # encoding the text and map each character to an integer and vice versa

    # We create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # Encode the text
    encoded = np.array([char2int[ch] for ch in text])

    # Showing the first 100 encoded characters
    # encoded[:100]
                          
    # Define and print the net
    n_hidden=512
    n_layers=2

    net = CharLSTM(chars, n_hidden, n_layers, train_on_gpu=train_on_gpu)
    print(net)

    # Declaring the hyperparameters
    batch_size = 128
    seq_length = 100
    n_epochs = 20 # start smaller if you are just testing initial behavior

    # print(net)

    # path = 'rnn_15_epoch_all_data.net'
    # load_model(net, path)

    net.train()

    # train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=2, train_on_gpu=train_on_gpu)

    # Saving the model
    model_name = 'rnn_20_epoch_all_data_notags.net'

    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': chars,
                  'int2char': int2char,
                  'char2int': char2int}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)


    # model_name = 'rnn_10_epoch_all_data.net'

    # load_model(net, model_name)
        
    # Generating new text
    print(sample(net, lines=20, prime='an ', top_k=5, train_on_gpu=train_on_gpu))

    # TODO
    '''
    training set perplexity
        - should be low
        - entropy: bits of information given by sampling
        - perplexity: 2^(entropy)
            - selecting among PPL equal probable items
        - check probabilities of words in the training set
            - should be low probability

    try word based model
        - pre trained encoding

    evaluation
        - take utterance of training set and use that as bleu score calculation
        - can see whether or not my scores are good, or if BLEU scores are bad 

    read through different unicode encodings and see if any work

    try and data mine past AC games for other data
        - look at encodings
        - look at contexts

    train two models, personality and context
    '''