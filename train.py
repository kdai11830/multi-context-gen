# Importing libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import platform
import xml.etree.ElementTree as ET
import html
from itertools import chain
import shutil
import re
import random

from models.char_lstm import CharLSTM
from models.w2v_lstm import EmbeddingLSTM

from utils import load_data, clean_data, load_data_edited, tokenize_data, one_hot_encode, get_batches, get_pretrained_weights, create_embedding

from eval import predict, sample


# Declaring the train method
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=2, train_on_gpu=False):

    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_tokens = len(net.tokens)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            # DON'T ONE HOT ENCODE FOR EMBEDDINGS
            #x = one_hot_encode(x, n_tokens)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([Variable(each).data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            # print('is cuda: ', is_cuda)
            output, h = net(Variable(inputs), h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    # x = one_hot_encode(x, n_tokens)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


if __name__ == '__main__':

    plf = platform.system()
    rootdir = ''
    split_key = ''

    if 'Windows' in plf:
        # rootdir = 'C:\\Users\\Kevin Dai\\Desktop\\Schoolwork\\Thesis\\data\\3-classes-replaced-tags'
        rootdir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\multi-context-gen'
        split_key = '\\'
    else:
        rootdir = '/n/home13/kdai/multi-context-gen'
        split_key = '/'

    # Check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU!')
    else: 
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    # data = load_data(rootdir, n_files=0)
    # data = clean_data(data)
    data = load_data_edited(rootdir + split_key + 'data' + split_key + '3-classes-replaced-tags')
    random.shuffle(data)
    print('data length: ', len(data))
    text = '\n'.join(data)
    print('text length: ', len(text))
    print('training W2V model...')
    w2v_model_path = rootdir + split_key + 'models' + split_key + 'trained' + split_key + 'w2v_embeddings.model'
    w2v_model, word_tokens = get_pretrained_weights(model_path=None, data=data)
    weights = w2v_model.wv
    # word_tokens = list(chain.from_iterable(word_tokens))
    word_tokens = list(w2v_model.wv.vocab.keys())
    print('word tokens length: ', len(word_tokens))

    print('--------EXAMPLE--------\n',text[:1000],'\n------------------------\n')

    # print(text)
    # encoding the text and map each character to an integer and vice versa

    # We create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to integers
    tokens = tuple(set(word_tokens))
    int2token = dict(enumerate(tokens))
    token2int = {ch: ii for ii, ch in int2token.items()}
    # print(w2v_model.wv.index2word)
    token2idx = {token: token_index for token_index, token in enumerate(w2v_model.wv.index2word)} 
    # print(token2idx)

    print(word_tokens[:300])
    # Encode the text
    encoded = np.array([token2idx[token] for token in word_tokens])
    print('encoded shape: ',encoded.shape)

    # Showing the first 100 encoded characters
    # encoded[:100]
                          
    # Define and print the net
    n_hidden=512
    n_layers=2

    # net = CharLSTM(chars, n_hidden, n_layers, train_on_gpu=train_on_gpu)
    net = EmbeddingLSTM(tokens, n_hidden, n_layers, train_on_gpu=train_on_gpu, w2v=w2v_model)
    print(net)

    # Declaring the hyperparameters
    batch_size = 64 # smaller => less memory used
    seq_length = 100
    n_epochs = 20 # start smaller if you are just testing initial behavior

    # print(net)

    # path = 'rnn_15_epoch_all_data.net'
    # load_model(net, path)

    net.train()

    # train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.1, print_every=2, train_on_gpu=train_on_gpu)

    # Saving the model
    model_name = 'models' + split_key + 'rnn_' + str(n_epochs) + '_epoch_all_data_w2v.net'

    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': tokens,
                  'int2token': int2token,
                  'token2int': token2int}
    # save w2v path
    if isinstance(net, EmbeddingLSTM):
    	checkpoint['w2v_model_path'] = w2v_model_path

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)


    # model_name = 'rnn_10_epoch_all_data.net'

    # load_model(net, model_name)
        
    # Generating new text
    print(sample(net, lines=20, prime='an', top_k=5, train_on_gpu=train_on_gpu))