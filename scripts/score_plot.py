import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import operator

# returns dict with all the contexts as keys and scores as values
def read_scores(path):
	with open(path, 'r') as f:
		s = f.read()
		scores = json.loads(s)
		# print(scores)
		for key in scores.keys():
			if ' <' in key:
				new_key = re.sub(' <', '', key)
				scores[new_key] = scores.pop(key)
		return scores

# finds line counts in every context
# will be useful as x-axis for plots to show n_line to BLEU score correlation
# returns dict mapping contexts to n_lines
def count_lines(rootdir):
	lines_counts = {}
	for root, dirs, files in os.walk(rootdir):
		for file in files:
			filename = os.fsdecode(file)
			context_lst = filename[:-4].split('_')
			context = context_lst[0] + ' ' + context_lst[1] + ' '
			if len(context_lst) > 2:
				context += '_'.join(context_lst[2:]) + ' '

			with open(rootdir + '\\' + filename, 'r') as f:
				lines = f.readlines()
				n_lines = len(lines)
				lines_counts[context] = n_lines
	return lines_counts

# param: list of dicts that have scores for
#	baseline, topk, topp, topk+topp
# bar plot comparing the four scores
def plot(scores_lst, n_samples, line_counts=None):
	sorted_scores = {}

	if line_counts is not None:
		# sort by line counts
		sorted_scores = sorted(line_counts.items(), key=operator.itemgetter(1))
	else:
		# sort by baseline scores:
		sorted_scores = sorted(scores_lst[0].items(), key=operator.itemgetter(1))

	inc = len(sorted_scores) // n_samples

	x_labels = []
	scores = [[],[],[],[]]
	score_types = ['baseline', 'top-k', 'top-p', 'top-k + top-p']

	for i in range(0, len(sorted_scores), inc):
		x_labels.append(sorted_scores[i][0])

	for label in x_labels:
		for i in range(4):
			scores[i].append(scores_lst[i][label])

	colors = ['tab:blue','tab:orange','tab:green','tab:red']
	gap = .9 / len(scores)

	fig, ax = plt.subplots()

	ax.set_ylabel('BLEU Scores')
	ax.set_title('BLEU Score Performance')
	ax.set_xticklabels(x_labels)
	ax.set_xticks(np.arange(len(x_labels)))

	x = np.arange(len(x_labels))

	for i, row in enumerate(scores):
		ax.bar(x + (i-1.5) * gap, row, width=gap, color=colors[i%len(colors)], label=score_types[i])

	ax.legend()
	fig.tight_layout()
	plt.show()


def get_max_scores(scores_lst):
	score_types = ['baseline', 'top-k', 'top-p', 'top-k + top-p']
	scores = {}

	for i in range(len(scores_lst)):
		sorted_scores = sorted(scores_lst[i].items(), key=operator.itemgetter(1), reverse=True)
		j = 0
		while sorted_scores[j][1] == 1.0:
			j += 1
		scores[score_types[i]] = sorted_scores[j]

	return scores


def main():
	scores_lst = []
	rootdirs = ['C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\scores_baseline.txt',
		'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\scores_topk.txt',
		'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\scores_topp.txt',
		'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\scores_topk_topp.txt']


	for rootdir in rootdirs:
		scores_lst.append(read_scores(rootdir))

	data_dir = 'C:\\Users\\kdai1\\Desktop\\Schoolwork\\Thesis\\Edited\\3-class'
	line_counts = count_lines(data_dir)

	n_samples = 15
	plot(scores_lst, n_samples, line_counts=None)

	print(get_max_scores(scores_lst))

if __name__ == '__main__':
	main()

