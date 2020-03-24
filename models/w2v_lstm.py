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

from utils import load_data, clean_data, load_data_edited, tokenize_data, one_hot_encode, get_batches, get_pretrained_weights, create_embedding


##################################### EMBEDDING MODEL ##########################################


class EmbeddingLSTM(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001, train_on_gpu=False, w2v=None):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating word dictionaries
        self.tokens = tokens
        self.w2v = w2v
        self.weights = w2v.wv
        self.int2token = w2v.wv.index2word
        self.token2int = {token: token_index for token_index, token in enumerate(w2v.wv.index2word)}
        
        # create embedding layer
        self.embedding, embedding_dim = create_embedding(torch.from_numpy(w2v.wv.syn0).float(), True)

        #define the LSTM
        self.lstm = nn.LSTM(embedding_dim, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)

        #define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        #define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.tokens))

        self.train_on_gpu = train_on_gpu

        # use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print("training on", torch.cuda.device_count(), "GPUS...")
            self.lstm = nn.DataParallel(self.lstm)

    def forward(self, x, hidden):
                
        #get the outputs and the new hidden state from the lstm
        # print('is cuda: ', hidden[1].is_cuda)
        # print(hidden)
        x = x.type(torch.LongTensor)
        print(type(x))
        print(x.size())
        if self.train_on_gpu:
            x = x.to(torch.device('cuda'))
            
        # print(self.embedding(x))
        print(self.embedding(x).size())
        r_output, hidden = self.lstm(self.embedding(x), hidden)
        
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
