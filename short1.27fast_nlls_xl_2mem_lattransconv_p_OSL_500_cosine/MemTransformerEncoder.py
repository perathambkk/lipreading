import torch
import torch.nn as nn
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

	def __init__(self, encoder_layer, fast_encoder_layer, num_layers, memory=None, memory2=None, norm=None, input_dims=[2048, 256], n_classes=500):
		super(MemTransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.flayers = _get_clones(fast_encoder_layer, num_layers)
		self.num_layers = num_layers
		self.memory = memory
		self.memory2 = memory2
		self.input_dims = input_dims
		self.norm = norm
		# self.norm1 = nn.LayerNorm(input_dims[0])
		self.norm2 = nn.LayerNorm(input_dims[1])
		self.fuse_layer = nn.Linear(input_dims[0]+input_dims[1], self.input_dims[0])
		# self.fuse_layer = nn.Bilinear(input_dims[0], input_dims[1], self.input_dims[0])
		self.fuse_layer_norm = nn.Sequential(
				nn.BatchNorm1d(self.input_dims[0]),
				nn.ReLU(True),
				)
		self.fuse_layers = _get_clones(self.fuse_layer, num_layers-1)
		self.fuse_normlayers = _get_clones(self.fuse_layer_norm, num_layers-1)
		# self.fuse_pool = nn.AvgPool3d(kernel_size=[2,1,1], stride=[2,1,1], ceil_mode=True)
		self.fuse_conv = nn.Conv3d(1,1,kernel_size=[1,1,1], stride=[1,1,1],padding=[1,0,0])
		self.fuse_convs = _get_clones(self.fuse_conv, num_layers-1)
		return
		

	def forward(self, src, mask=None, src_key_padding_mask=None):
		"""Pass the input through the endocder layers in turn.

		Args:
			src: the sequnce to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output0 = src[0]
		output1 = src[1]
		
		batch_size = output1.size(1)
		fuse_ap = nn.AdaptiveAvgPool3d((23, batch_size, self.input_dims[1]))

		for i in range(self.num_layers):
			output0 = self.layers[i](output0, src_mask=mask,
									src_key_padding_mask=src_key_padding_mask)
			output1 = self.flayers[i](output1, src_mask=mask,
									src_key_padding_mask=src_key_padding_mask)
			if i < self.num_layers - 1:
				moutput1 = output1.unsqueeze(0).unsqueeze(0)
				# moutput1 = self.fuse_pool(moutput1)
				moutput1 = self.fuse_convs[i](moutput1)
				moutput1 = fuse_ap(moutput1)
				moutput1 = moutput1.squeeze(0).squeeze(0)
				# moutput1 = torch.mean(output1,1)
				# moutput1 = moutput1.unsqueeze(2).repeat(1, 1, output0.size(2))
				
				output0 = self.fuse_layers[i](torch.cat([output0, moutput1],2))
				# output0 = self.fuse_layers[i](output0, moutput1)
				output0 = output0.permute(1,2,0)
				output0 = self.fuse_normlayers[i](output0)
				output0 = output0.permute(2,0,1)
			if self.memory and i == self.num_layers - 2:
				output0 = output0 + self.memory(output0)
			if self.memory2 and i == self.num_layers - 2:
				output1 = output1 + self.memory2(output1)

		if self.norm:
			output0 = self.norm(output0)
			output1 = self.norm2(output1)

		return [output0, output1]
		
