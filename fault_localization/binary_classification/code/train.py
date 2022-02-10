import os
import sys
import torch
import time
import pickle
import random
from gensim.models.word2vec import Word2Vec
from model import BinaryClassifier


def get_batch(x, y, idx, bs):
	x_batch = x[idx: idx+bs]
	y_batch = y[idx: idx+bs]
	return torch.LongTensor(x_batch), torch.LongTensor(y_batch)


def load_from_file(file_path):
	with open(file_path, "rb") as file:
		return pickle.load(file)


def print_parameter_statistics(model):
	total_num = [p.numel() for p in model.parameters()]
	trainable_num = [p.numel() for p in model.parameters() if p.requires_grad]
	print("Total parameters: {}".format(sum(total_num)))
	print("Trainable parameters: {}".format(sum(trainable_num)))


if __name__ == "__main__":
	fix_pattern = sys.argv[1]
	print("Fix pattern: {}".format(fix_pattern))
	root = "../data/{}/".format(fix_pattern)

	train_x = load_from_file(os.path.join(root, "train/x_w2v_.pkl"))
	val_x = load_from_file(os.path.join(root, "val/x_w2v_.pkl"))
	test_x = load_from_file(os.path.join(root, "test/x_w2v_.pkl"))
	
	train_y = load_from_file(os.path.join(root, "train/y_.pkl"))
	val_y = load_from_file(os.path.join(root, "val/y_.pkl"))
	test_y = load_from_file(os.path.join(root, "test/y_.pkl"))

	pretrain_vectors = load_from_file(os.path.join(root, "vectors.pkl"))

	HIDDEN_DIM = 50
	EPOCHS = 30
	BATCH_SIZE = 64
	LABELS = 2
	USE_GPU = True
	MAX_TOKENS = pretrain_vectors.shape[0]
	EMBEDDING_DIM = pretrain_vectors.shape[1]

	print("MAX_TOKENS: {}".format(MAX_TOKENS))
	print("EMBEDDING_DIM: {}".format(EMBEDDING_DIM))
	print("HIDDEN_DIM: {}".format(HIDDEN_DIM))
	print("BATCH_SIZE: {}".format(BATCH_SIZE))

	model = BinaryClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS, LABELS, pretrain_vectors)

	if USE_GPU:
		model.cuda()

	parameters = model.parameters()
	optimizer = torch.optim.Adam(parameters, lr=0.001)
	print("Optimizer: {}".format(type(optimizer).__name__))
	loss_function = torch.nn.CrossEntropyLoss()
	print("Loss function: {}".format(type(loss_function).__name__))
	print_parameter_statistics(model)

	train_loss_ = []
	val_loss_ = []

	train_acc_ = []
	val_acc_ = []

	train_pre_ = []
	val_pre_ = []

	train_recall_ = []
	val_recall_ = []

	train_f1_ = []
	val_f1_ = []

	best_val_acc = 0.0

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
		total_loss = 0.0
		total = 0
		total_tp = 0.0
		total_fp = 0.0
		total_fn = 0.0
		
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
			total += len(train_inputs)
			total_loss += loss.item() * len(train_inputs)
			total_tp += (((predicted == 0).long() + (train_labels == 0).long()) == 2).sum().item()
			total_fp += (((predicted == 0).long() + (train_labels == 1).long()) == 2).sum().item()
			total_fn += (((predicted == 1).long() + (train_labels == 0).long()) == 2).sum().item()

		train_loss_.append(total_loss / total)
		train_acc_.append(total_acc / total)
		precision = total_tp / (total_tp + total_fp + 1e-6)
		train_pre_.append(precision)
		recall = total_tp / (total_tp + total_fn + 1e-6)
		train_recall_.append(recall)
		train_f1_.append(2 * precision * recall / (precision + recall + 1e-6))

		# validating phase
		model.eval()
		total_acc = 0.0
		total_loss = 0.0
		total = 0
		total_tp = 0.0
		total_fp = 0.0
		total_fn = 0.0
		
		i = 0
		while i < len(val_x):
			batch = get_batch(val_x, val_y, i, BATCH_SIZE)
			val_inputs, val_labels = batch
			if USE_GPU:
				val_inputs = val_inputs.cuda()
				val_labels = val_labels.cuda()
			i += BATCH_SIZE
			output = model(val_inputs)
			loss = loss_function(output, val_labels)
			
			_, predicted = torch.max(output.data, 1)
			total_acc += (predicted == val_labels).sum().item()
			total += len(val_inputs)
			total_loss += loss.item() * len(val_inputs)
			total_tp += (((predicted == 0).long() + (val_labels == 0).long()) == 2).sum().item()
			total_fp += (((predicted == 0).long() + (val_labels == 1).long()) == 2).sum().item()
			total_fn += (((predicted == 1).long() + (val_labels == 0).long()) == 2).sum().item()

		val_loss_.append(total_loss / total)
		val_acc_.append(total_acc / total)
		precision = total_tp / (total_tp + total_fp + 1e-6)
		val_pre_.append(precision)
		recall = total_tp / (total_tp + total_fn + 1e-6)
		val_recall_.append(recall)
		val_f1_.append(2 * precision * recall / (precision + recall + 1e-6))

		# save model
		if val_acc_[-1] > best_val_acc:
			model_save_dir = "./model_save/{}/".format(fix_pattern)
			if not os.path.exists(model_save_dir):
				os.makedirs(model_save_dir)
			torch.save(model.state_dict(), os.path.join(model_save_dir, "model_params.pkl"))
			torch.save(model, os.path.join(model_save_dir, "model.pkl"))
			best_val_acc = val_acc_[-1]
			print("="*30 + "\n" + "Saving model: epoch_{}".format(epoch + 1))

		end_time = time.time()
		print("[Epoch: %3d/%3d]\n"
			  "Training Loss: %.5f,\t\tValidation Loss: %.5f\n"
			  "Training Acc: %.5f,\t\tValidation Acc: %.5f\n"
			  "Training Pre: %.5f,\t\tValidation Pre: %.5f\n"
			  "Training Recall:%.5f,\tValidation Recall: %.5f\n"
			  "Training F1-score: %.5f,\tValidation F1-score: %.5f\n"
			  "Time Cost: %.3f s"
			  % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
				 train_acc_[epoch], val_acc_[epoch],
				 train_pre_[epoch], val_pre_[epoch],
				 train_recall_[epoch], val_recall_[epoch],
				 train_f1_[epoch], val_f1_[epoch],
				 end_time - start_time))

	# testing phase
	total_acc = 0.0
	total_loss = 0.0
	total = 0
	total_tp = 0.0
	total_fp = 0.0
	total_fn = 0.0

	model.load_state_dict(torch.load("./model_save/{}/model_params.pkl".format(fix_pattern)))
	model.eval()

	i = 0
	while i < len(test_x):
		batch = get_batch(test_x, test_y, i, BATCH_SIZE)
		test_inputs, test_labels = batch
		if USE_GPU:
			test_inputs = test_inputs.cuda()
			test_labels = test_labels.cuda()
		i += BATCH_SIZE
		output = model(test_inputs)
		
		_, predicted = torch.max(output.data, 1)
		total_acc += (predicted == test_labels).sum().item()
		total += len(test_inputs)
		total_tp += (((predicted == 0).long() + (test_labels == 0).long()) == 2).sum().item()
		total_fp += (((predicted == 0).long() + (test_labels == 1).long()) == 2).sum().item()
		total_fn += (((predicted == 1).long() + (test_labels == 0).long()) == 2).sum().item()
	precision = total_tp / (total_tp + total_fp + 1e-6)
	recall = total_tp / (total_tp + total_fn + 1e-6)
	f1 = 2 * precision * recall / (precision + recall + 1e-6)
	print("Testing Acc: %.5f" % (total_acc / total))
	print("Testing Pre: %.5f" % precision)
	print("Testing Recall: %.5f" % recall)
	print("Testing F1-score: %.5f" % f1)
