import os
from eval import sample, predict
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from gensim.models import Word2Vec
import json
import random

from models.char_lstm import CharLSTM
from models.w2v_lstm import EmbeddingLSTM

rootdir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\rnn_4_epoch_all_data_w2v.net'

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    # net.load_state_dict(checkpoint['state_dict'])
    # net.eval()
    return checkpoint

def eval_bleu_scores(net, trials=5, generate_samples=True):
	datapath = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-class'
	references = []
	scores = {}
	for root, dirs, files in os.walk(datapath):
		for file in files:
			with open(datapath + '\\' + file, 'r') as f:
				data = f.readlines()
			for line in data:
				references.append(word_tokenize(line))

			context = ' '.join(references[0][0:3]) + ' '
			# print(context)

			score = 0
			for i in range(trials):
				if generate_samples:
					generated = word_tokenize(sample(net, stop_after=300, prime=context, top_k=5, top_p=0.9, temperature=0.7).split('\n')[0])
					score += sentence_bleu(references, generated)
				else:
					generated = random.choice(references)
					new_references = references[:]
					new_references.remove(generated)
					score += sentence_bleu(new_references, generated)				
			score /= trials
			print(context, '| score:', score)
			scores[context] = score
			references = []
	return scores


device = torch.device('cpu')

checkpoint = load_checkpoint(rootdir)
tokens = checkpoint['tokens']
n_layers = checkpoint['n_layers']
n_hidden = checkpoint['n_hidden']
w2v_model_path = checkpoint['w2v_model_path']
# net = CharRNN(tokens, n_hidden, n_layers)
net = EmbeddingLSTM(tokens, n_hidden=n_hidden, n_layers=n_layers, w2v=Word2Vec.load(w2v_model_path))


net.load_state_dict(checkpoint['state_dict'])
net.eval()
    
# Generating new text
# print(sample(net, lines=50, stop_after=500,prime='GE FreeA Memory ', top_k=5, train_on_gpu=False).split('\n')[0])
print(sample(net, stop_after=2000, prime='BO Ev Xmas ', top_k=5, top_p=0.9, temperature=0.7, train_on_gpu=False))

# Calculate BLEU scores
'''scores = eval_bleu_scores(net, trials=3, generate_samples=True)
scores_file = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\scores_topk_topp.txt'
with open(scores_file, 'w') as f:
	f.write(json.dumps(scores))'''

