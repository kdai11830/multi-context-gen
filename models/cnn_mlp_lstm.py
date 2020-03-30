import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import platform

import re
import random

import utils

################################## MODELS ##################################

class CharCNN(nn.Module):

	'''
	Args:
	length: len of sentence/word
	input_dim: dimension of input, should be matrix of char embeddings (i.e. 2)
	output_dim: dimension of cnn output, should be feature matrix (i.e. 2)
	kernels: list of kernel widths

	'''
	def __init__(self, length, input_dim, feature_maps, kernels):

		super().__init__()
		self.length =  length
		self.input_dim = input_dim
		self.feature_maps = feature_maps
		self.kernels = kernels

		cnn_layers = []
		for i in range(len(kernels)):
			reduced_l = length - kernels[i] + 1
			conv = nn.Conv2d(input_dim, feature_maps[i], w)
			maxpool = nn.MaxPool2d(reduced_l)
			tanh = nn.Tanh()
			cnn_layers.append(conv, maxpool, tanh)

		self.layers = nn.Sequential(*cnn_layers)

	def forward(self, x):
		# input shape is (batch_size, length, input_dim)
		out = self.layers(x)


class HighwayMLP(nn.Module):

	def __init__(self, size, n_layers=1, f=nn.ReLU()):
		super().__init__()
		self.size = size
		self.n_layers = n_layers
		self.bias = bias
		self.f = f

		#layers
		self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(n_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(n_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(n_layers)])

    def forward(self, x):

    	out = x
    	for layer in range(self.n_layers):
    		gate = F.sigmoid(self.gate[layer](out))
    		nonlinear = self.f(self.nonlinear[layer](out))
    		linear = self.linear[layer](out)

    		out = gate * nonlinear + (1 - gate) + linear

    	return out

class CNN_MLP_LSTM(nn.Module):

	def __init__(self, hidden_size, n_layers, drop_prob, n_words, word_embed_size, n_chars, char_embed_size, feature_maps, kernels, length, n_highway=0):
		super().__init__()

		self.train_on_gpu = train_on_gpu

		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.drop_prob = drop_prob
		self.n_words = n_words
		self.word_embed_size = word_embed_size
		self.n_chars = n_chars
		self.char_embed_size = char_embed_size
		self.feature_maps = feature_maps
		self.kernels = kernels
		self.length = length
		self.n_highway = n_highway

		# declare layers
		self.charCNN = CharCNN(length, char_embed_size, feature_maps, kernels)

		input_size_L = torch.sum(torch.Tensor(feature_maps))
		if n_highway > 0:
			self.highwayMLP = HighwayMLP(input_size_L, n_highway)

		self.lstm = nn.LSTM(len(self.tokens), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
		self.dropout = nn.Dropout(drop_prob)
		self.fc = nn.Linear(n_hidden, len(self.tokens))

	def forward(self, x, hidden):

		out = self.charCNN(x)

		if self.n_highway > 0:
			out = self.highwayMLP(out)

		r_output, hidden = self.lstm(out, hidden)
        
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