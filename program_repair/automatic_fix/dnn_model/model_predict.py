import javalang
import os
import pickle
import re
import string
import torch
import numpy as np
import sys
import pickle
from gensim.models.word2vec import Word2Vec


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
		if (end_index - start_index) > 0 and (end_index - start_index + 1) <= token_length_for_reserve:
			padding_length = token_length_for_reserve - (end_index - start_index + 1)
			pre_padding_length = padding_length // 3 * 2
			post_padding_length = padding_length - pre_padding_length
			if start_index >= pre_padding_length and len(token_seq) - end_index - 1 >= post_padding_length:
				return token_seq[start_index - pre_padding_length: end_index + 1 + post_padding_length]
			elif start_index < pre_padding_length:
				return token_seq[0: token_length_for_reserve]
			elif len(token_seq) - end_index - 1 < post_padding_length:
				return token_seq[len(token_seq) - token_length_for_reserve: len(token_seq)]
		else:
			return token_seq[start_index: start_index + token_length_for_reserve]


def get_token_seq_big_vocab(source_path):
	with open(source_path, "r", encoding="utf-8") as file:
		content = file.read()
		tokens = javalang.tokenizer.tokenize(content)
		token_seq = []
		for token in tokens:
			tmp_token = solve_camel_and_underline(token.value)
			token_seq += tmp_token
		return token_seq


def get_x(content, v2v_path, padding_length):
	content = cut_data(content, padding_length)
	word2vec = Word2Vec.load(v2v_path).wv
	vocab = word2vec.vocab
	max_token = word2vec.syn0.shape[0]
	normal = [vocab[token].index if token in vocab else max_token for token in content]
	if len(normal) <= padding_length:
		normal += [max_token] * (padding_length - len(content))
	return normal


def model_predict(token_seq, model_path, w2v_path, padding_length):
	x = get_x(token_seq, w2v_path, padding_length)
	x = [x]
	x = torch.LongTensor(x)
	model = torch.load(model_path, map_location=lambda storage, loc: storage)
	model.eval()
	predict_result = model(x)
	rank = predict_result.topk(11, 1, True, True).indices
	rank = rank.cpu().detach().numpy().tolist()
	return rank


if __name__ == "__main__":
	buggy_version = sys.argv[1]
	current_dir = os.path.dirname(__file__)
	source_path = os.path.join(current_dir, "../method_tmp/{}.txt".format(buggy_version))
	with open(os.path.join(current_dir, "version_info.pkl"), "rb") as file:
		version_info = pickle.load(file)
	if buggy_version in version_info:
		model_path = os.path.join(current_dir, "model_save_d4j/{}/model.pkl".format(version_info[buggy_version]))
	else:
		model_path = ps.path.join(current_dir, "model_save_d4j/1/model.pkl")
	w2v_path = os.path.join(current_dir, "w2v_32")
	padding_length = 400
	token_seq = get_token_seq_big_vocab(source_path)	
	rank = model_predict(token_seq, model_path, w2v_path, padding_length)
	rank = [[str(factor + 1) for factor in record] for record in rank]
	
	for record in rank:
		record += ["12", "13", "14", "15"]
		print(" ".join(record))

