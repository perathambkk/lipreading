import torch
import copy

def _get_clones(module, N):
	return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MemTransformerEncoder(torch.nn.Module):
	"""TransformerEncoder is a stack of N encoder layers

	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		>>> src = torch.rand(10, 32, 512)
		>>> out = transformer_encoder(src)
	"""

	def __init__(self, encoder_layer, num_layers, memory, norm=None):
		super(MemTransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm
		self.memory = memory

	def forward(self, src, mask=None, src_key_padding_mask=None):
		"""Pass the input through the endocder layers in turn.

		Args:
			src: the sequnce to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output = src

		for i in range(self.num_layers):
			output = self.layers[i](output, src_mask=mask,
									src_key_padding_mask=src_key_padding_mask)
			if i == self.num_layers - 2:
				output = output + self.memory(output)

		if self.norm:
			output = self.norm(output)

		return output
		