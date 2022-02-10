from torch import nn
import torch


class MLP(nn.Module):

	def __init__(self):
		super(MLP, self).__init__()
		
		self.mlp_semantic = nn.Linear(11, 3)
		self.mlp_all_features = nn.Linear(10, 10)
		
		self.output_layer = nn.Linear(10, 2)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, inputs):
		spectrum = inputs[:, 0:3]
		mutation = inputs[:, 3:7]
		semantic = inputs[:, 7:]
		
		semantic = self.dropout(self.activation(self.mlp_semantic(semantic)))
		all_features = torch.cat([spectrum, mutation, semantic], dim = -1)
		all_features = self.dropout(self.activation(self.mlp_all_features(all_features)))
		out = self.output_layer(all_features)
		return out
