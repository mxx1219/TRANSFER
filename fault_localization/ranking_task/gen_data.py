import os
import pickle
import numpy as np

data_path = "../binary_classification/d4j_data/"

spectrum_path = os.path.join(data_path, "spectrum.pkl")
with open(spectrum_path, "rb") as file:
	spectrum = pickle.load(file)

mutation_path = os.path.join(data_path, "mutation.pkl")
with open(mutation_path, "rb") as file:
	mutation = pickle.load(file)

semantic_path = os.path.join(data_path, "semantic.pkl")
with open(semantic_path, "rb") as file:
	semantic = pickle.load(file)

with open("faulty_statement_index.pkl", "rb") as file:
	faulty_stmts = pickle.load(file)

# parse labels (0 for faulty and 1 for non-faulty)
label = {}
for version in spectrum:
	current_label = [1] * len(spectrum[version])
	for faulty_index in faulty_stmts[version]:
		current_label[faulty_index - 1] = 0
	label[version] = current_label

# parse data
data = {}
for version in spectrum:
	data[version] = []
	# init
	for i in range(len(spectrum[version])):
		data[version].append([])
	# spectrum features (3-dim: ochiai, dstar, tarantula)
	current_spectrum = np.array(spectrum[version])
	for i in range(3):
		scores = current_spectrum[:, i].tolist()
		sorted_scores = sorted(scores, reverse=True)
		ranking_scores = [sorted_scores[::-1].index(score) / len(scores) for score in scores]
		for i, score in enumerate(ranking_scores):
			data[version][i].append(score)
	# mutation features (4-dim: different granularities with ochiai algorithm)
	current_mutation = np.array(mutation[version])
	for i in range(4):
		scores = current_mutation[:, i].tolist()
		sorted_scores = sorted(scores, reverse=True)
		ranking_scores = [sorted_scores[::-1].index(score) / len(scores) for score in scores]
		for i, score in enumerate(ranking_scores):
			data[version][i].append(score)
	# semantic features (11-dim: fix patterns from tbar)
	current_semantic = np.array(semantic[version])
	for i in range(11):
		scores = current_semantic[:, i].tolist()
		sorted_scores = sorted(scores, reverse=True)
		ranking_scores = [sorted_scores[::-1].index(score) / len(scores) for score in scores]
		for i, score in enumerate(ranking_scores):
			data[version][i].append(score)

# save data and label
with open("./run_model/x.pkl", "wb") as file:
	pickle.dump(data, file)
with open("./run_model/y.pkl", "wb") as file:
	pickle.dump(label, file)
