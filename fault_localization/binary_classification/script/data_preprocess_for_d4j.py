import os
import javalang
import pickle
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
import string
import random
import sys


def solve_camel_and_underline(token):
	if token.isdigit():
		return [token]
	else:
		p = re.compile(r'([a-z]|\d)([A-Z])')
		sub = re.sub(p, r'\1_\2', token).lower()
		sub_tokens = sub.split("_")
		tokens = re.sub(" +", " ", " ".join(sub_tokens)).strip()
		final_token = []
		for factor in tokens.split(" "):
			final_token.append(factor.rstrip(string.digits))
		return final_token


def cut_data(token_seq, token_length_for_reserve):
	if len(token_seq) <= token_length_for_reserve:
		return token_seq
	else:
		start_index = token_seq.index("rank2fixstart")
		end_index = token_seq.index("rank2fixend")
		assert end_index > start_index
		length_of_annotated_statement = end_index - start_index + 1
		if length_of_annotated_statement <= token_length_for_reserve:
			padding_length = token_length_for_reserve - length_of_annotated_statement
			# give 2/3 padding space to content before annotated statement
			pre_padding_length = padding_length // 3 * 2
			# give 1/3 padding space to content after annotated statement
			post_padding_length = padding_length - pre_padding_length
			if start_index >= pre_padding_length and len(token_seq) - end_index - 1 >= post_padding_length:
				return token_seq[start_index - pre_padding_length: end_index + 1 + post_padding_length]
			elif start_index < pre_padding_length:
				return token_seq[:token_length_for_reserve]
			elif len(token_seq) - end_index - 1 < post_padding_length:
				return token_seq[len(token_seq) - token_length_for_reserve:]
		else:
			return token_seq[start_index: start_index + token_length_for_reserve]


if __name__ == "__main__":

	current_pattern = sys.argv[1]
	pattern_list = ["InsertMissedStmt",
					"InsertNullPointerChecker",
					"MoveStmt",
					"MutateConditionalExpr",
					"MutateDataType",
					"MutateLiteralExpr",
					"MutateMethodInvExpr",
					"MutateOperators",
					"MutateReturnStmt",
					"MutateVariable",
					"RemoveBuggyStmt"]
	if current_pattern not in pattern_list:
		print("The fix pattern specified by the argument dost not exist.")
		exit(-1)
	
	print("Current fix pattern: {}".format(current_pattern))
	
	# Path declaration
	print("Path declaration")
	data_dir = "../data/{}/".format(current_pattern)
	d4j_data_dir = "../d4j_data/"

	# Parameters declaration
	print("Parameters declaration")
	token_length_for_reserve = 400

	# Loading vocab and vectors
	print("Loading vocab and vectors")
	with open(os.path.join(data_dir, "vocab.pkl"), "rb") as file:
		vocab_dict = pickle.load(file)
	with open(os.path.join(data_dir, "vectors.pkl"), "rb") as file:
		vectors = pickle.load(file)

	# Data preprocessing for defects4j data
	print("Data preprocessing for defects4j data")
	index_oov = vocab_dict["OOV"]
	index_padding = vocab_dict["PADDING"]
	
	
	with open(os.path.join(d4j_data_dir, "src_code.pkl"), "rb") as file:
		src_code = pickle.load(file)
	with open(os.path.join(d4j_data_dir, "checker_info.pkl"), "rb") as file:
		checker_info = pickle.load(file)
	d4j_w2v_ = {}
	for project in src_code:
		samples = []
		for method in src_code[project]:
			method = method.strip()
			tokens = javalang.tokenizer.tokenize(method)
			token_seq = []
			for token in tokens:
				if isinstance(token, javalang.tokenizer.String):
					tmp_token = ["stringliteral"]
				else:
					tmp_token = solve_camel_and_underline(token.value)
				token_seq += tmp_token
			token_seq = cut_data(token_seq, token_length_for_reserve)
			normal_record = [vocab_dict[token] if token in vocab_dict else index_oov for token in token_seq]
			if len(normal_record) < token_length_for_reserve:
				normal_record += [index_padding] * (token_length_for_reserve - len(normal_record))
			samples.append(normal_record)
		d4j_w2v_[project] = samples
	with open(os.path.join(data_dir, "d4j_w2v_.pkl"), "wb") as file:
		pickle.dump(d4j_w2v_, file)

	print("Done")
