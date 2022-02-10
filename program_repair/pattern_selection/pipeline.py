import os
import re
from gensim.models.word2vec import Word2Vec
import pickle
import random
import string
import javalang

class Pipeline:
	def __init__(self, ratio, negative_sampling, root):
		self.ratio = ratio
		self.root = root
		self.seq_length = 400
		self.negative_sampling = negative_sampling
		self.size = 32
		self.patterns = list()
		self.patterns.append("InsertNullPointerChecker")
		self.patterns.append("MoveStmt")
		self.patterns.append("MutateDataType")
		self.patterns.append("MutateLiteralExpr")
		self.patterns.append("MutateOperators")
		self.patterns.append("MutateVariable")
		self.patterns.append("InsertMissedStmt")
		self.patterns.append("MutateConditionalExpr")
		self.patterns.append("MutateMethodInvExpr")
		self.patterns.append("MutateReturnStmt")
		self.patterns.append("RemoveBuggyStmt")


	def parse_source(self):
		corpus = []
		labels = []
		with open("../dataset_pr.pkl", "rb") as file:
			dataset = pickle.load(file)
		for pattern in self.patterns:
			pattern_samples = dataset[pattern][:]
			random.seed(666)
			random.shuffle(pattern_samples)
			pattern_samples = pattern_samples[:self.negative_sampling]
			for sample in pattern_samples:
				token_seq = self.parse_token_sequence(sample)
				corpus.append(token_seq)
			labels += [self.patterns.index(pattern)] * len(pattern_samples)
		self.dump_to_file(corpus, os.path.join(self.root, "x_data.pkl"))
		self.dump_to_file(labels, os.path.join(self.root, "y_data.pkl"))


	def parse_d4j_source(self):
		corpus = []
		src_path = os.path.join(self.root, "d4j_data/src_data")
		label_file_path = os.path.join(self.root, "d4j_data/pattern.txt")
		files = os.listdir(src_path)
		for i in range(1, len(files) + 1):
			with open(os.path.join(src_path, "{}.txt".format(i)), "r") as file:
				sample = file.read()
			token_seq = self.parse_token_sequence(sample)
			corpus.append(token_seq)
		with open(os.path.join(label_file_path), "r") as label_file:
			labels = [[int(label) for label in line.strip().split(" ")] for line in label_file.readlines()]
		
		
		def check_or_create(path):
			if not os.path.exists(path):
				os.mkdir(path)

		d4j_path = self.root + 'd4j/'
		check_or_create(d4j_path)
		self.dump_to_file(corpus, d4j_path + "x_.pkl")
		self.dump_to_file(labels, d4j_path + "y_.pkl")

	
	def parse_token_sequence(self, sample):
		tokens = javalang.tokenizer.tokenize(sample)
		token_seq = []
		for token in tokens:
			if isinstance(token, javalang.tokenizer.String):
				tmp_token = ["stringliteral"]
			else:
				tmp_token = self.solve_camel_and_underline(token.value)
			token_seq += tmp_token
		token_seq = self.cut_data(token_seq)
		return token_seq

	
	def solve_camel_and_underline(self, token):
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
	
	
	def cut_data(self, token_seq):
		if len(token_seq) <= self.seq_length:
			return token_seq
		else:
			start_index = token_seq.index("rank2fixstart")
			end_index = token_seq.index("rank2fixend")
			assert end_index > start_index
			length_of_annotated_statement = end_index - start_index + 1
			if length_of_annotated_statement <= self.seq_length:
				padding_length = self.seq_length - length_of_annotated_statement
				# give 2/3 padding space to content before annotated statement
				pre_padding_length = padding_length // 3 * 2
				# give 1/3 padding space to content after annotated statement
				post_padding_length = padding_length - pre_padding_length
				if start_index >= pre_padding_length and len(token_seq) - end_index - 1 >= post_padding_length:
					return token_seq[start_index - pre_padding_length: end_index + 1 + post_padding_length]
				elif start_index < pre_padding_length:
					return token_seq[:self.seq_length]
				elif len(token_seq) - end_index - 1 < post_padding_length:
					return token_seq[len(token_seq) - self.seq_length:]
			else:
				return token_seq[start_index: start_index + self.seq_length]
	
	
	def dump_to_file(self, obj, file_path):
		with open(file_path, "wb") as file:
			pickle.dump(obj, file)

	
	def load_from_file(self, file_path):
		with open(file_path, "rb") as file:
			return pickle.load(file)

	
	def split_data(self):
		corpus = self.load_from_file(os.path.join(self.root, "x_data.pkl"))
		labels = self.load_from_file(os.path.join(self.root, "y_data.pkl"))
		data_num = len(corpus)
		random.seed(666)
		random.shuffle(corpus)
		random.seed(666)
		random.shuffle(labels)
		ratios = [int(r) for r in self.ratio.split(':')]
		train_split = int(ratios[0] / sum(ratios) * data_num)
		val_split = train_split + int(ratios[1] / sum(ratios) * data_num)
		x_train = corpus[:train_split]
		y_train = labels[:train_split]
		x_val = corpus[train_split:val_split]
		y_val = labels[train_split:val_split]
		x_test = corpus[val_split:]
		y_test = labels[val_split:]

		def check_or_create(path):
			if not os.path.exists(path):
				os.mkdir(path)

		train_path = self.root + 'train/'
		check_or_create(train_path)
		self.dump_to_file(x_train, train_path + "x_.pkl")
		self.dump_to_file(y_train, train_path + "y_.pkl")

		val_path = self.root + 'val/'
		check_or_create(val_path)
		self.dump_to_file(x_val, val_path + "x_.pkl")
		self.dump_to_file(y_val, val_path + "y_.pkl")

		test_path = self.root + 'test/'
		check_or_create(test_path)
		self.dump_to_file(x_test, test_path + "x_.pkl")
		self.dump_to_file(y_test, test_path + "y_.pkl")

	
	def dictionary_and_embedding(self, input_file, size):
		self.size = size
		if not input_file:
			input_file = self.root + "train/x_.pkl"
		corpus = self.load_from_file(input_file)
		if not os.path.exists(self.root + 'train/embedding'):
			os.mkdir(self.root + 'train/embedding')

		from gensim.models.word2vec import Word2Vec
		w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=2, max_vocab_size=50000)
		w2v.save(self.root + 'train/embedding/w2v_' + str(self.size))

	
	def generate_normal_data(self, part):
		word2vec = Word2Vec.load(self.root + 'train/embedding/w2v_' + str(self.size)).wv
		vocab = word2vec.vocab
		max_token = word2vec.syn0.shape[0]

		corpus = self.load_from_file(self.root + part + "/x_.pkl")
		normal_corpus = []
		for record in corpus:
			normal_record = [vocab[token].index if token in vocab else max_token for token in record]
			if len(record) <= self.seq_length:
				# select another index for padding (different from oov index)
				normal_record += [max_token + 1] * (self.seq_length - len(record))
			else:
				normal_record = normal_record[:self.seq_length]
			normal_corpus.append(normal_record)
		self.dump_to_file(normal_corpus, self.root + part + "/x_w2v_.pkl")

	
	def run(self):
		print('parse source code...')
		self.parse_source()
		print('split data...')
		self.split_data()
		print('train word embedding...')
		self.dictionary_and_embedding(None, 32)
		print('generate normal data...')
		self.generate_normal_data('train')
		self.generate_normal_data('val')
		self.generate_normal_data('test')

	
	def run_d4j(self):
		print('parse d4j source code...')
		self.parse_d4j_source()
		print('generate d4j normal data...')
		self.generate_normal_data('d4j')


if __name__ == "__main__":
	pipeline = Pipeline("8:1:1", 10000, "data/")
	#pipeline.run()
	pipeline.run_d4j()
	print("done.")

