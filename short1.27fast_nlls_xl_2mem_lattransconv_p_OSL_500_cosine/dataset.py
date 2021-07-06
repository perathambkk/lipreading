# encoding: utf-8
import os
import cv2
import glob
import numpy as np
import torch
from dt_transform import *

min_scale = [122]
max_scale = [122] #146
crop_size = [112]

def load_file(filename):
	arrays = np.load(filename)['data']
	# arrays = np.stack([cv2.resize(arrays[_], (122, 122))
	#                  for _ in range(29)], axis=0)
	# arrays = arrays / 255.
	arrays = arrays - [0.45, 0.45, 0.45]
	arrays = arrays / [0.225, 0.225, 0.225]
	return arrays


class MyDataset():
	def __init__(self, folds, path):
		self.folds = folds
		self.path = path
		with open('../label_sorted.txt') as myfile:
			self.data_dir = myfile.read().splitlines()
		self.filenames = glob.glob(os.path.join(self.path, '*', self.folds, '*.npz'))
		self.list = {}
		for i, x in enumerate(self.filenames):
			target = x.split('/')[-3] 
			for j, elem in enumerate(self.data_dir):
				if elem == target:
					self.list[i] = [x]
					self.list[i].append(j)
		print('Load {} part'.format(self.folds))

	def __getitem__(self, idx):
		inputs = load_file(self.list[idx][0])
		inputs = torch.from_numpy(inputs)
		inputs = inputs.float().permute(0, 3, 1, 2)
		if self.folds == 'train':
			inputs = random_short_side_scale_jitter(
					inputs, min_scale, max_scale
				)
			inputs = random_crop(inputs, crop_size[0])
			inputs = horizontal_flip(0.5, inputs)
		elif self.folds == 'val' or self.folds == 'test':
			inputs = random_short_side_scale_jitter(
					inputs, min_scale, max_scale
				)
			inputs = uniform_crop(inputs, crop_size[0], 1)
		else:
			raise Exception('the split doesn\'t exist')
		inputs = inputs.float().permute(1, 0, 2, 3)
		inputs = inputs.numpy()
		labels = self.list[idx][1]
		return inputs, labels

	def __len__(self):
		return len(self.filenames)
