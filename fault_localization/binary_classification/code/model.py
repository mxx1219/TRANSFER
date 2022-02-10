from torch import nn
import torch
import torch.nn.functional as F


class BinaryClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, pretrained_weight=None):
		super(BinaryClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.label_size = label_size
		self.activation = torch.tanh
		self.num_layers = 1

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#if pretrained_weight is not None:
		#	self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
		self.embedding.weight.requires_grad = True
		self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
		self.decoder = nn.Linear(hidden_dim * 2, self.label_size)

	def forward(self, inputs):
		embeddings = self.embedding(inputs)
		lstm_out, hidden = self.encoder(embeddings)
		lstm_out = torch.transpose(lstm_out, 1, 2)
		out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
		out = self.decoder(out)
		return out
