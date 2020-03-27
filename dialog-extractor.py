import os
import re
import codecs
import shutil
import xml.etree.ElementTree as ET
# import html
from collections import Counter
from itertools import islice
import platform

import nltk
from nltk.tokenize import word_tokenize

import random

plf = platform.system()
rootdir = ''
split_key = ''

if 'Windows' in plf:
    # rootdir = 'C:\\Users\\Kevin Dai\\Desktop\\Schoolwork\\Thesis\\Talk'
    rootdir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\multi-context-gen'
    split_key = '\\'
else:
    rootdir = '/n/home13/kdai/multi-context-gen'
    split_key = '/'



six_entity_prefix = [
	'&#xE;&#x5;&#xF;&#x2;',
	'&#xE;&#x11;&#x1;&#x2;',
	'&#xE;&#x5;&#x10;&#x2;',
	'&#xE;&#x5;&#x11;&#x2;',
	'&#xE;&#x0;&#x2;&#x2;',
	'&#xE;&#xD;&#x8;&#x2;'
]

def add_xml_extension(src):
	for item in os.listdir(src):
		s = os.path.join(src, item)
		if os.path.isdir(s):
			add_xml_extension(s)
		elif s.endswith('.kup'):
			shutil.copy(s, s + '.xml')

# add_xml_extension(rootdir)

def window(seq, n=2):
	it = iter(seq)
	result = tuple(islice(it, n))
	if len(result) == n:
		yield result
	for elem in it:
		result = result[1:] + (elem,)
		yield result


# longest common subsequence helper function using dynamic programming
def lcs(x, y):
	m = len(x)
	n = len(y)

	L = [[None]*(n+1) for i in range(m+1)]
	for i in range(m+1):
		for j in range(n+1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif x[i-1] == y[j-1]:
				L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i-1][j], L[i][j-1])

	return L[m][n]


# extract numeric character entities from data string
def extract_nce(s, entities):

	# start by replacing known entities
	if len(entities) > 0:
		for i in range(len(entities)):
			s = s.replace(entities[i], '<'+str(i)+'>')

	# extract remaining entities
	started = False
	in_entity = False
	in_encoded = False
	new_entities = []
	temp = ''
	entity_count = 0
	entity_len = 4
	i = 0
	while i < len(s):
		# print(s[i].encode('unicode_escape').decode('utf-8'))
		decoded = s[i].encode('unicode_escape').decode('utf-8')
		if not started:
			# started a known cue
			if decoded == '&':
				started = True
				in_entity = True
				temp += decoded

		else:
			# within the group of four for basic entities
			if entity_count < entity_len:
				temp += decoded
				# print(temp)
				if in_entity and decoded == ';':
					in_entity = False
					entity_count += 1
				elif not in_entity and decoded == '&':
					in_entity = True
					if in_encoded:
						entity_count += 1
						in_encoded = False
				elif not in_entity and decoded != '&':
					in_encoded = True

				# if key for 6 entity string
				if any(pfx in temp for pfx in six_entity_prefix):
					entity_len = 6

				# if key for 8 entity pause string
				elif '&#xE;&#x7;&#x0;&#x4;' in temp:
					entity_len = 8

			# finished with 4 or 6 entities
			else:
				# print(temp)
				# print(s[i:i+10])

				# check if it is 6 entity substring
				'''if '&#x0;&#x0;' in s[i:i+10]:
					temp += s[i:i+10]
					i += 10'''

				# reset entity length to default
				entity_len = 4

				# print(temp)

				new_entities.append(temp)
				temp = ''
				started = False
				in_entity = False
				entity_count = 0
				entity_len = 4

				decoded = s[i].encode('unicode_escape').decode('utf-8')

				# started a new nce
				# might start with \\t or \\n
				if decoded == '&': # or s(s[i-1] == ';' and decoded == '\\t' or decoded == '\\n'):
					started = True
					in_entity = True
					temp += decoded
				elif i != len(s) - 1:
					if s[i+1] == '&' and s[i] != '.' and s[i] != ',' and s[i] != ' ' and s[i] != '!' and s[i] != '?' and s[i] != '\n' and s[i] != '\"':
						started = True
						in_entity = True
						temp += decoded
						entity_count += 1

		i += 1

	# do final pass to combine pause entities (8 tags)
	'''
	temp_entities = []
	pause_tag = '&#xE;&#x7;&#x0;&#x4;'
	i = 0
	while i < len(entities) - 1:
		if new_entities[i] == pause_tag:
			temp = new_entities[i] + new_entities[i+1]
			temp_entities.append(temp)
			i += 1
		else:
			temp_entities.append(new_entities[i])
		i += 1
	'''

	# replace new entities found and add to existing entities
	last_ind = len(entities)
	entities = entities + new_entities
	# print(len(entities))
	for i in range(last_ind, len(entities), 1):
		s = s.replace(entities[i], ' <'+str(i)+'> ')

	return s, entities

# TODO: TEST THIS
def remove_tags(rootdir=None, target_dir=None, replace_proper_nouns=False):
	if rootdir is None:
		rootdir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes'
	if target_dir is None:
		target_dir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes-tag-removed'

	for file in os.listdir(rootdir):
		filename = os.fsdecode(file)
		print(filename)
		with open(file, 'r') as rf:
			lines = rf.readlines()
			with open(target_dir + '\\' + filename, 'w') as wf:
				for line in lines:
					s = re.sub(r'<[0-9]+>', ' ', line)
					if replace_proper_nouns:
						s = re.sub(r',[ ]+([.!?]{1})', ', NOUN_PLACEHOLDER\1', s)
					s = re.sub(r'[ ]+', ' ', s)
					wf.write(s + '\n')
	

def replace_tags(rootdir=None, target_dir=None, replace_proper_nouns=False):
	if rootdir is None:
		rootdir = 'C:\\Users\\Kevin Dai\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes'
	if target_dir is None:
		target_dir = 'C:\\Users\\Kevin Dai\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes-replaced-tags'

	for file in os.listdir(rootdir):
		filename = os.fsdecode(file)
		print(filename)
		with open(rootdir + '\\' + filename, 'r') as rf:
			lines = rf.readlines()
			with open(target_dir + '\\' + filename, 'w') as wf:
				for s in lines:
					# replace all tags and tag strings
					s = re.sub(r'(<[0-9]+>)+', ' $ENTITY ', s)
					# replace ellipses with pause entities in between
					s = re.sub(r'( \$ENTITY .)+ \$ENTITY ', '... ',  s)
					s = re.sub(r'[ ]+', ' ', s)
					wf.write(s)

	print('DONE')
	return



def combine_duplicate_contexts(rootdir=None, target_dir=None, context_num=0):
	if context_num not in [0,1,2]:
		print('INVALID CONTEXT NUMBER')
		return

	if rootdir is None:
		rootdir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes'
	if target_dir is None:
		target_dir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-classes-combined-context'

	for file in os.listdir(rootdir):
		filename = os.fsdecode(file)
		context = filename[:-4].split('_')


def check_encodings():
	test_file = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Talk\\AN_3P_An_umsbt\\00000000.msbt.kup.xml'
	codecs = ['ascii', 'cp037', 'cp932', 'euc-jp', 'iso2022-jp', 'utf-16le', 'utf-7', 'utf-8']
	for c in codecs:
		try:
			with open(test_file, 'r', encoding=c) as tree:
				# raw = html.unescape(tree.readlines())
				lines = tree.readlines()
				print(c)
				print(lines[:20])
				print('----------------------------------------\n----------------------------------------')
		except UnicodeDecodeError:
			print('UnicodeDecodeError reached with', c, ' encoding')


# combine all data into one file
# separate into labels and data points
def compile_data(rootdir, target_dir):
	data = []
	directory = os.fsencode(rootdir)
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		print(filename)
		with open(rootdir + split_key + filename, 'r') as rf:
			for line in rf.readlines():
				data.append(line)

	random.shuffle(data)

	labels = []
	utterances = []
	for line in data:
		words = word_tokenize(line)
		label = ' '.join(words[:3])
		utterance = ' '.join(words[3:])
		labels.append(label)
		utterances.append(utterance)

	# split to train, test, and validation
	test_idx = len(data) // 2
	test_label = labels[:test_idx]
	test_data = utterances[:test_idx]

	train_idx = len(data) - int(len(data) * 0.08)
	train_label = labels[test_idx:train_idx]
	train_data = utterances[test_idx:train_idx]

	val_label = labels[train_idx:]
	val_data = utterances[train_idx:]

	# write all to files
	with open(target_dir + split_key + 'test.src', 'w') as wlabels:
		for label in test_label:
			wlabels.write(label + '\n')
	with open(target_dir + split_key + 'train.src', 'w') as wlabels:
		for label in train_label:
			wlabels.write(label + '\n')
	with open(target_dir + split_key + 'valid.src', 'w') as wlabels:
		for label in val_label:
			wlabels.write(label + '\n')
	with open(target_dir + split_key + 'test.tgt', 'w') as wdata:
		for utterance in test_data:
			utterance = utterance.replace('$ ENTITY', '$ENTITY')
			wdata.write(utterance + '\n')
	with open(target_dir + split_key + 'train.tgt', 'w') as wdata:
		for utterance in train_data:
			utterance = utterance.replace('$ ENTITY', '$ENTITY')
			wdata.write(utterance + '\n')
	with open(target_dir + split_key + 'valid.tgt', 'w') as wdata:
		for utterance in val_data:
			utterance = utterance.replace('$ ENTITY', '$ENTITY')
			wdata.write(utterance + '\n')

def get_vocab(rootdir):
	script = []
	with open(rootdir, 'r', encoding='utf-8') as f:
		for line in f.readlines():
			words = word_tokenize(line)
			script += words

	vocab = list(set(script))
	with open(rootdir + '.txt', 'w', encoding='utf-8') as wf:
		for term in vocab:
			wf.write(term + '\n')


def main():
	minimum_size = 3
	maximum_size = 8

	all_entities = []

	print('start')
	i = 0
	for subdir, dirs, files in os.walk(rootdir):
		# get personality and context info from file name
		personality = ''
		context_1 = ''
		context_2 = ''
		if subdir.split(split_key)[-1] != 'Talk':
			filename = subdir.split(split_key)[-1].split('_')
			# print(filename)
			del filename[-1]
			personality = filename[0]
			context_1 = filename[1]
			context_2 = '_'.join(filename[2:])
			new_file = 'Edited' + split_key + '_'.join(filename) + '.txt'

		for file in files:
			if 'xml' in file and '1' not in file and '2' not in file: # only want 00000000.kup because the rest aren't english
				print(i)
				i += 1

				f = open(new_file, 'w')

				with open(subdir + split_key + file, 'r', encoding='utf-8') as tree:
					raw = html.unescape(tree.readlines())

					# prune data for usable information
					data = []
					cur = personality + ' ' + context_1 + ' ' + context_2 + ' '
					started = False
					for line in raw:
						if '<original>' in line and '</original>' in line:
							cur += line
							cur = cur.replace('<original>', '')
							cur = cur.replace('</original>', '')
							data.append(cur)
							cur = personality + ' ' + context_1 + ' ' + context_2 + ' '
						elif '<original>' in line and not started:
							started = True
							cur += line
						elif '</original>' in line:
							started = False
							cur += line
							cur = cur.replace('<original>', '')
							cur = cur.replace('</original>', '')
							data.append(cur)
							cur = personality + ' ' + context_1 + ' ' + context_2 + ' '
						elif started:
							cur += line

						
					# print(data)

					# USE extract_nce METHOD TO WRITE STRINGS TO FILE TO SAVE MEMORY
					for i in range(len(data)):
						s, all_entities = extract_nce(data[i], all_entities)
						s = re.sub('&#x[0-9a-eA-E]{1,2};', '', s)
						s = s.replace('\t', '')
						s = s.replace('\n', ' ')
						s = re.sub(r'[^\x00-\x7F]+','', s)
						s = re.sub(' +',' ', s).strip()
						f.write(s + '\n')
						# print(s)
						# print()
					f.close()



if __name__ == '__main__':
	# main()
	# check_encodings()
	# replace_tags()
	# compile_data(rootdir + split_key + 'data' + split_key + '3-classes-replaced-tags', rootdir + split_key + 'data' + split_key + 'data-compiled')
	get_vocab(rootdir + split_key + 'data' + split_key + 'data-compiled' + split_key + 'data.txt.bpe')