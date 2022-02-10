import os
import sys
import torch
import time
import numpy as np
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from model import BinaryClassifier


def load_from_file(file_path):
	with open(file_path, "rb") as file:
		return pickle.load(file)


if __name__ == "__main__":
	fix_patterns = ["InsertMissedStmt", "InsertNullPointerChecker", "MoveStmt", "MutateConditionalExpr",
			"MutateDataType", "MutateLiteralExpr", "MutateMethodInvExpr", "MutateOperators",
			"MutateReturnStmt", "MutateVariable", "RemoveBuggyStmt"]
	out_semantic_features = {}
	for fix_pattern in fix_patterns:
		print("Fix pattern: {}".format(fix_pattern))
		root = "../data/{}/".format(fix_pattern)
		checker_info_path = "../d4j_data/checker_info.pkl"

		d4j_data = load_from_file(os.path.join(root, "d4j_w2v_.pkl"))
		checker_info = load_from_file(checker_info_path)
		pretrain_vectors = load_from_file(os.path.join(root, "vectors.pkl"))

		HIDDEN_DIM = 50
		LABELS = 2
		USE_GPU = True
		MAX_TOKENS = pretrain_vectors.shape[0]
		EMBEDDING_DIM = pretrain_vectors.shape[1]

		model = BinaryClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS, LABELS, pretrain_vectors)

		if USE_GPU:
			model.cuda()

		# predicting for defects4j data (11-dimensional semantic features)
		model.load_state_dict(torch.load("./model_save/{}/model_params.pkl".format(fix_pattern)))
		model.eval()

		for project in d4j_data:
			input_samples = d4j_data[project]
			input_samples = torch.LongTensor(input_samples)
			if USE_GPU:
				input_samples = input_samples.cuda()
			if project not in out_semantic_features:
				out_semantic_features[project] = []
			output = model(input_samples)
			output = torch.softmax(output, dim=-1)
			output = output[:, 0].cpu().tolist()
			for index, checker_flag in enumerate(checker_info[project]):
				if fix_patterns.index(fix_pattern) == 0:
					if checker_flag[fix_patterns.index(fix_pattern)] == 0:
						out_semantic_features[project].append([0])
					else:
						out_semantic_features[project].append([output[index]])
				else:
					if checker_flag[fix_patterns.index(fix_pattern)] == 0:
						out_semantic_features[project][index].append(0)
					else:
						out_semantic_features[project][index].append(output[index])
	with open("../d4j_data/semantic.pkl", "wb") as file:
		pickle.dump(out_semantic_features, file)
