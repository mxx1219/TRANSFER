import torch
import time
import os
import sys
import numpy as np
import pickle
import random
from model import MultiClassifier
from gensim.models.word2vec import Word2Vec


def get_batch(x, y, idx, bs):
	x_batch = x[idx: idx+bs]
	y_batch = y[idx: idx+bs]
	return torch.LongTensor(x_batch), torch.LongTensor(y_batch)

def load_from_file(file_path):
	with open(file_path, "rb") as file:
		return pickle.load(file)

def parse_dataset(x_data, y_data, version_path, part):
	# get all d4j versions which may be fixed by 11 pre-defined fix patterns
	with open(version_path, "r") as file:
		content = file.readlines()
	content = [line.strip() for line in content]
	version_list = []
	version_set = []
	for line in content:
		current_version = line[:line.rindex("_")]
		version_list.append(current_version)
		if current_version not in version_set:
			version_set.append(current_version)
	print("All d4j versions which may be fixed: {}".format(len(version_set)))
	
	# 10-fold cross validation (split defects4j versions)
	step = (len(version_set) - 1) // 10 + 1
	random.seed(666)
	random.shuffle(version_set)
	test_part = version_set[(part-1)*step: part*step]
	train_part = []
	for version in version_set:
		if version not in test_part:
			train_part.append(version)
	d4j_model_save_path = "./model_save_d4j/{}/".format(part)
	if not os.path.exists(d4j_model_save_path):
		os.makedirs(d4j_model_save_path)
	with open(os.path.join(d4j_model_save_path, "test_versions.txt"), "w") as file:
		file.write("\n".join(test_part))
	
	# 10-fold cross validation (split bugs according to d4j versions)
	# one d4j version could contain more than 1 bugs
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	test_bugs = []
	for index, version in enumerate(version_list):
		if version in train_part:
			train_x.append(x_data[index])
			train_y.append(y_data[index].index(1))
		elif version in test_part:
			test_x.append(x_data[index])
			test_y.append(y_data[index])
			test_bugs.append(content[index])
	print("train bugs: {}\ttest bugs: {}".format(len(train_x), len(test_x)))
	return train_x, train_y, test_x, test_y, test_bugs
	

def print_parameter_statistics(model):
	total_num = [p.numel() for p in model.parameters()]
	trainable_num = [p.numel() for p in model.parameters() if p.requires_grad]
	print("Total parameters: {}".format(sum(total_num)))
	print("Trainable parameters: {}".format(sum(trainable_num)))


if __name__ == "__main__":
	root = "data/"
	version_path = os.path.join(root, "d4j_data/versions.txt")
	x_data = load_from_file(os.path.join(root, "d4j/x_w2v_.pkl"))
	y_data = load_from_file(os.path.join(root, "d4j/y_.pkl"))
	part = int(sys.argv[1]) # range [1,10]
	train_x, train_y, test_x, test_y, test_bugs = parse_dataset(x_data, y_data, version_path, part)

	word2vec = Word2Vec.load(os.path.join(root, "train/embedding/w2v_32")).wv
	embeddings = np.zeros((word2vec.syn0.shape[0] + 2, word2vec.syn0.shape[1]), dtype="float32")

	TOKEN_LENGTH = 400
	HIDDEN_DIM = 80
	EPOCHS = 30
	BATCH_SIZE = 8
	LABELS = 11
	USE_GPU = True
	MAX_TOKENS = word2vec.syn0.shape[0] + 2
	EMBEDDING_DIM = word2vec.syn0.shape[1]

	print("MAX_TOKENS:{}".format(MAX_TOKENS))
	print("EMBEDDING_DIM:{}".format(EMBEDDING_DIM))
	print("HIDDEN_DIM: {}".format(HIDDEN_DIM))
	print("BATCH_SIZE: {}".format(BATCH_SIZE))

	model = MultiClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS, LABELS, embeddings)
	model.load_state_dict(torch.load("./model_save/model_params.pkl"))
	
	if USE_GPU:
		model.cuda()

	parameters = model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=0.0001)
	print("Optimizer: {}".format(type(optimizer).__name__))
	loss_function = torch.nn.CrossEntropyLoss()
	print("Loss function: {}".format(type(loss_function).__name__))
	print_parameter_statistics(model)
		
	train_loss_ = []
	train_acc_ = []
	train_mrr_ = []

	print('Start training ...')
	for epoch in range(EPOCHS):
		start_time = time.time()

		# shuffle data before training for each epoch
		random.seed(666)
		random.shuffle(train_x)
		random.seed(666)
		random.shuffle(train_y)

		# training phase
		model.train()
		total_acc = 0.0
		total_mrr = 0.0
		total_loss = 0.0
		total = 0
		
		i = 0
		while i < len(train_x):
			batch = get_batch(train_x, train_y, i, BATCH_SIZE)
			train_inputs, train_labels = batch
			if USE_GPU:
				train_inputs = train_inputs.cuda()
				train_labels = train_labels.cuda()
			i += BATCH_SIZE
			model.zero_grad()
			output = model(train_inputs)
			loss = loss_function(output, train_labels)
			loss.backward()
			optimizer.step()
			
			_, predicted = torch.max(output.data, 1)
			total_acc += (predicted == train_labels).sum().item()
			total_mrr += (1 / (torch.where((output.topk(output.shape[1], 1, True, True).indices.t() - train_labels.t()).t() == 0)[1].float() + 1)).sum().item()
			total += len(train_inputs)
			total_loss += loss.item() * len(train_inputs)

		train_loss_.append(total_loss / total)
		train_acc_.append(total_acc / total)
		train_mrr_.append(total_mrr / total)
		
		if epoch + 1 == EPOCHS:
			model_save_dir = "./model_save_d4j/"
			torch.save(model.state_dict(), os.path.join(model_save_dir, "{}/model_params.pkl".format(part)))
			torch.save(model, os.path.join(model_save_dir, "{}/model.pkl".format(part)))
		
		end_time = time.time()
		print('[Epoch: %3d/%3d] Training Loss: %.4f,'
				' Training Acc: %.3f, Training Mrr: %.3f, Time Cost: %.3f s'
			% (epoch + 1, EPOCHS, train_loss_[epoch],
			train_acc_[epoch], train_mrr_[epoch], end_time - start_time))
			
	# testing phase
	model.load_state_dict(torch.load("./model_save_d4j/{}/model_params.pkl".format(part)))
	model.eval()
	fix_count = 0
	cannot_fix_list = []
		
	i = 0
	while i < len(test_x):
		batch = get_batch(test_x, test_y, i, BATCH_SIZE)
		i += BATCH_SIZE
		test_inputs, test_labels = batch
		if USE_GPU:
			test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()

		output = model(test_inputs)
			
		neg_inf = torch.full_like(test_labels, -1e10).float()
		earliest_appear_labels = torch.gather(test_labels, 1, torch.argmax(torch.where(test_labels==0, neg_inf, output.data), axis=1).unsqueeze(-1))
		#fix_count += (earliest_appear_labels == 1).sum().item()
		cannot_fix_list += (np.where((earliest_appear_labels == -1).cpu().numpy())[0] + i - BATCH_SIZE).tolist()
		
	cannot_fix_list = [test_bugs[factor] for factor in cannot_fix_list]
	print("Cannot fix list: {}".format(",".join(cannot_fix_list)))

