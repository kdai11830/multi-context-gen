# Importing libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from nltk.tokenize import word_tokenize

import os
import platform
import shutil
import re
import random

from utils import load_data, clean_data, load_data_edited, tokenize_data, one_hot_encode, get_batches, get_pretrained_weights, create_embedding

from models.char_lstm import CharLSTM
from models.w2v_lstm import EmbeddingLSTM

# Defining a method to generate the next token
def predict(net, token, h=None, top_k=None, train_on_gpu=False):
    ''' Given a token, predict the next token.
        Returns the predicted token and the hidden state.
    '''
            
    # tensor inputs
    x = np.array([[net.token2int[token]]])
    # x = one_hot_encode(x, len(net.tokens))
    inputs = torch.from_numpy(x)
            
    if(train_on_gpu):
        inputs = inputs.cuda()
            
    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the token probabilities
    p = F.softmax(out, dim=1).data
    if(train_on_gpu):
        p = p.cuda() # move to cpu
            
    # get top tokens
    if top_k is None:
        top_ch = np.arange(len(net.tokens))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.cpu().numpy().squeeze()
    
    # select the likely next token with some element of randomness
    p = p.cpu().numpy().squeeze()
    token = np.random.choice(top_ch, p=p/p.sum())
            
    # return the encoded value of the predicted token and the hidden state
    return net.int2token[token], h
            
# Declaring a method to generate new text
def sample(net, lines=1, stop_after=5000, prime='the', top_k=None, train_on_gpu=False):
            
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
        
    net.eval() # eval mode
        
    # First off, run through the prime tokens
    tokens = []
    if isinstance(net, CharLSTM):
        tokens = [ch for ch in prime]
        h = net.init_hidden(1)
        for ch in prime:
            token, h = predict(net, ch, h, top_k=top_k)
            tokens.append(token)
    elif isinstance(net, EmbeddingLSTM):
        tokens = word_tokenize(prime)
        print(tokens)
        h = net.init_hidden(1)
        for token in tokens:
            token, h = predict(net, prime, h, top_k=top_k)
            tokens.append(token)
    else:
        print('SOME SHIT WENT WRONG')
        return


    # Now pass in the previous token and get a new one
    # Stop either after we get wanted number of lines or until we reach upper limit
    ii = 0
    stop = 0
    while ii < lines and stop < stop_after:
        token, h = predict(net, tokens[-1], h, top_k=top_k, train_on_gpu=train_on_gpu)
        tokens.append(token)
        if tokens[-2:-1] == '\n':
            ii += 1
        stop += 1


    return ''.join(tokens)


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