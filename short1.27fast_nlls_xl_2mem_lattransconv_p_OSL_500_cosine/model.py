# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from slowfast.models import model_builder, head_helper
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu

from PKMlayer import HashingMemory, AttrDict
from MemTransformerEncoderLayer import MemTransformerEncoderLayer
from MemTransformerEncoder import MemTransformerEncoder

device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")



class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_len=5000, offset=0):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.offset = offset

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[self.offset:self.offset+x.size(0), :]
		return self.dropout(x)
		

class TransformerEnc(nn.Module):
	def __init__(self, input_size, num_heads, num_layers, num_classes, every_frame=True):
		super(TransformerEnc, self).__init__()
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.every_frame = every_frame
		### Memories ###
		params = AttrDict({
				"sparse": False,
				"k_dim": 128,
				"heads": 4,
				"knn": 32,
				"n_keys": 168,  # the memory will have (n_keys ** 2) values
				"query_batchnorm": True,
				"input_dropout": 0,
				"query_dropout": 0,
				"value_dropout": 0.1,
			})
		self.memory = HashingMemory(input_dim=input_size[0], output_dim=input_size[0], params=params)
		params = AttrDict({
				"sparse": False,
				"k_dim": 128,
				"heads": 4,
				"knn": 32,
				"n_keys": 50,  # the memory will have (n_keys ** 2) values
				"query_batchnorm": True,
				"input_dropout": 0,
				"query_dropout": 0,
				"value_dropout": 0.1,
			})
		self.memory2 = HashingMemory(input_dim=input_size[1], output_dim=input_size[1], params=params)
		self.norm = nn.LayerNorm(input_size[0])
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size[0], nhead=self.num_heads)
		self.fencoder_layer = nn.TransformerEncoderLayer(d_model=input_size[1], nhead=self.num_heads)
		self.transformer_encoder = MemTransformerEncoder(self.encoder_layer, self.fencoder_layer, num_layers=self.num_layers, \
			memory=self.memory, memory2=self.memory2, norm=self.norm)
		# self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
		# self.fc = nn.Linear(input_size, num_classes)
		return

	def forward(self, x):
		out = self.transformer_encoder(x)
		out[0] = out[0].transpose(0, 1)
		out[1] = out[1].transpose(0, 1)
		# if self.every_frame:
		# 	out = self.fc(out)  # predictions based on every time step
		# else:
		# 	out = self.fc(out[:, -1, :])  # predictions based on last time-step
		# return out
		return out


class Lipreading(nn.Module):
	def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True):
		super(Lipreading, self).__init__()
		self.mode = mode
		self.inputDim = inputDim
		self.hiddenDim = hiddenDim
		self.nClasses = nClasses
		self.frameLen = frameLen
		self.every_frame = every_frame
		self.nLayers = 6
		self.nHeads = 8
		self.p_encoder_dropout = 0.1

		# slowfast net
		cfg = get_cfg()
		np.random.seed(cfg.RNG_SEED)
		torch.manual_seed(cfg.RNG_SEED)
		
		# Load config from cfg.
		cfg_file = 'configs/LRW/SLOWFAST_NLN_8x8_R50.yaml'
		cfg.merge_from_file(cfg_file)
		self.slowfast_model = model_builder.build_model(cfg)
		self.slowfast_model = torch.load('models/SLOWFAST_8x8_1.27R50.pt')

		# cfg.TRAIN.CHECKPOINT_TYPE = 'caffe2'
		# cfg.TRAIN.CHECKPOINT_FILE_PATH = 'models/SLOWFAST_8x8_R50.pkl'
		# cu.load_checkpoint(
		#	cfg.TRAIN.CHECKPOINT_FILE_PATH,
		#	self.slowfast_model,
		#	False,
		#	None,
		#	inflation=False,
		#	convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
		# )
		# pool_size = [[1, 1, 1], [1, 1, 1]]
		# width_per_group = cfg.RESNET.WIDTH_PER_GROUP
		# self.head = head_helper.ResNetBasicHead(
		#   dim_in=[
		#       width_per_group * 32,
		#       width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
		#   ],
		#   num_classes=500,
		#   pool_size=[
		#       [
		#           cfg.DATA.NUM_FRAMES 
		#           // cfg.SLOWFAST.ALPHA
		#           // pool_size[0][0] + 1,
		#           cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
		#           cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
		#       ],
		#       [
		#           cfg.DATA.NUM_FRAMES // pool_size[1][0],
		#           cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
		#           cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
		#       ],
		#   ],
		#   dropout_rate=cfg.MODEL.DROPOUT_RATE,
		# )
		# resnet
		# self.resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.inputDim)
		pool_size = [[1,4,4], [1,4,4]]
		for pathway in range(2):
			avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
			self.add_module("pathway{}_avgpool".format(pathway), avg_pool)
		# backend_conv
		input_dims = [2048, 256]
		kernel_strides = [[3,2],[5,2]]
		for pathway in range(2):
			backend_conv1 = nn.Sequential(
					nn.Conv1d(input_dims[pathway], 2*input_dims[pathway], kernel_strides[pathway][0], kernel_strides[pathway][1], 0, bias=False),
					nn.BatchNorm1d(2*input_dims[pathway]),
					nn.ReLU(True),
					nn.MaxPool1d(2, 2),
					nn.Conv1d(2*input_dims[pathway], 4*input_dims[pathway], kernel_strides[pathway][0], kernel_strides[pathway][1], 0, bias=False),
					nn.BatchNorm1d(4*input_dims[pathway]),
					nn.ReLU(True),
					)
			self.add_module("pathway{}_backend_conv1".format(pathway), backend_conv1)
		self.backend_conv2 = nn.Sequential(
				nn.Linear(4*sum(input_dims), self.inputDim),
				nn.BatchNorm1d(self.inputDim ),
				nn.ReLU(True),
				nn.Linear(self.inputDim , 500)
				)
		
		# self.p_encoder = PositionalEncoding(self.inputDim, self.p_encoder_dropout)
		# self.t_encoder = TransformerEnc(self.inputDim, self.nHeads, self.nLayers, self.nClasses, self.every_frame)
		self.inputDim1 = 2048
		self.inputDim2 = 256
		self.inputDim3 = 512
		self.p_encoder1 = PositionalEncoding(self.inputDim1, self.p_encoder_dropout, offset=3)
		self.t_encoder1 = TransformerEnc([self.inputDim1,self.inputDim2], self.nHeads, self.nLayers, self.nClasses, self.every_frame)
		self.p_encoder2 = PositionalEncoding(self.inputDim2, self.p_encoder_dropout)
		self.backend_tlinear = nn.Sequential(
				nn.Linear(sum(input_dims), self.inputDim),
				nn.BatchNorm1d(self.inputDim ),
				nn.ReLU(True),
				nn.Linear(self.inputDim , 500)
				)
		# self.backend_tlinear2 = nn.Sequential(
		# 		nn.Linear(2*self.inputDim3, self.inputDim),
		# 		nn.BatchNorm1d(self.inputDim ),
		# 		nn.ReLU(True),
		# 		nn.Linear(self.inputDim , 500)
		# 		)
		# initialize
		self._initialize_weights()

	def forward(self, x):
		# x = [x[0].contiguous(), x[1].contiguous()]
		x = self.slowfast_model(x)
		if self.mode == 'temporalConv':
			x[0] = self.pathway0_avgpool(x[0])
			x[0] = x[0].view(-1, 23, 2048)
			x[0] = x[0].permute(0, 2, 1)
			x[0] = self.pathway0_backend_conv1(x[0])
			x[0] = torch.mean(x[0], 2)

			x[1] = self.pathway1_avgpool(x[1])
			x[1] = x[1].view(-1, 29, 256)
			x[1] = x[1].permute(0, 2 ,1)
			x[1] = self.pathway1_backend_conv1(x[1])
			x[1] = torch.mean(x[1], 2)
			x = torch.cat(x, 1)

			x = self.backend_conv2(x)
			return x

		elif self.mode == 'backendSelfAttention' or self.mode == 'finetuneSelfAttention':
			x[0] = self.pathway0_avgpool(x[0])
			x[0] = x[0].view(-1, 23, 2048)
			x[0] = x[0].permute(1, 0, 2)			
			x[0] = self.p_encoder1(x[0])

			x[1] = self.pathway1_avgpool(x[1])
			x[1] = x[1].view(-1, 29, 256)
			x[1] = x[1].permute(1, 0, 2)			
			x[1] = self.p_encoder2(x[1])

			x = self.t_encoder1(x)
			
			x[0] = x[0].permute(0, 2, 1)
			x[0] = self.pathway0_backend_conv1(x[0])
			x[0] = torch.mean(x[0], 2)
			x[1] = x[1].permute(0, 2 ,1)
			x[1] = self.pathway1_backend_conv1(x[1])
			x[1] = torch.mean(x[1], 2)
			x = torch.cat(x, 1)

			x = self.backend_conv2(x)
			# x[0] = torch.mean(x[0], 1)
			# x[1] = torch.mean(x[1], 1)
			# x = (x[0] + x[1]) / 2.
			# x = torch.cat(x, 1)
			# x = self.backend_tlinear(x)
			return x
		else:
			return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


def lipreading(mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True):
	model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, every_frame=every_frame)
	return model

class MyDataParallel(torch.nn.DataParallel):
	"""
	Allow nn.DataParallel to call model's attributes.
	"""
	def __getattr__(self, name):
		try:
			return super().__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)
