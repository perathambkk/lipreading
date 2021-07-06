import torch
import copy
import numpy as np

from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import functional as F
from RprMultiheadAttention import MultiheadAttention

def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu
	else:
		raise RuntimeError("activation should be relu/gelu, not %s." % activation)
		
class MemTransformerEncoderLayer(torch.nn.Module):
	"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.

	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of intermediate layer, relu or gelu (default=relu).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> src = torch.rand(10, 32, 512)
		>>> out = encoder_layer(src)
	"""

	def __init__(self, d_model, nhead, frame_len=29, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(MemTransformerEncoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model)

		self.norm1 = LayerNorm(d_model)
		self.norm2 = LayerNorm(d_model)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)

		self.activation = _get_activation_fn(activation)

		# rpr encoding
		self.frame_len = frame_len
		self.k = 5
		self.create_rpr_table(self.k)
		self.pk_embed = nn.Embedding(2*self.k + 1, d_model)
		self.pv_embed = nn.Embedding(2*self.k + 1, d_model)

	def idx_clip(x, k):
		return max(-k, min(k, x))

	def create_rpr_table(self, k):
		self.rpr_table = np.zeros((self.frame_len, self.frame_len))
		for i in range(1, self.frame_len + 1):
			for j in range(1, self.frame_len + 1):
				self.rpr_table[i-1, j-1] = idx_clip(j-i, k) + k
		return

	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		r"""Pass the input through the endocder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2 = self.self_attn(src, src, src, posk, posv, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		if hasattr(self, "activation"):
			src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		else:  # for backward compatibility
			src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src
