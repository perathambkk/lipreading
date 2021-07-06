# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import pickle 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from model import *
from dataset import *
from ls_loss import *
from lr_scheduler import *
from th_transform import *
from noam_opt import *
import pytorch_warmup as warmup

import torch.multiprocessing as multiprocessing 
multiprocessing.set_start_method('spawn', force=True)

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")


def data_loader(args):
	dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'val', 'test']}
	dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers) for x in ['train', 'val', 'test']}
	dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
	print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
	return dset_loaders, dset_sizes


def reload_model(model, logger, path=""):
	if torch.cuda.is_available():
		logger.info('using GPU')
	if not bool(path):
		logger.info('train from scratch')
		return model
	else:
		model_dict = model.state_dict()
		pretrained_dict = torch.load(path)
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		logger.info('*** model has been successfully loaded! ***')
		return model


def showLR(optimizer):
	lr = []
	for param_group in optimizer.param_groups:
		lr += [param_group['lr']]
	return lr


def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, scheduler, warmup_scheduler, args, logger, use_gpu, save_path):
	if phase == 'val' or phase == 'test':
		model.eval()
	if phase == 'train':
		model.train()
	if phase == 'train':
		logger.info('-' * 10)
		logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
		logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

	running_loss, running_corrects, running_all = 0., 0., 0.
	# slow_idx = np.asarray(np.linspace(0,28,8), dtype=int)
	slow_idx = torch.tensor(range(3,26)).long()
	min_scale = [128]
	max_scale = [160]
	crop_size = [112]
			
	for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
		# inputs = inputs.float().permute(0, 1, 4, 2, 3)
		# if phase == 'train':
		# 	inputs = random_short_side_scale_jitter(
		# 		inputs, min_scale, max_scale
		# 	)
		# 	inputs = random_crop(inputs, crop_size[0])
		# 	inputs = horizontal_flip(0.5, inputs)
		# elif phase == 'val' or phase == 'test':
		# 	inputs = random_short_side_scale_jitter(
		# 		inputs, crop_size, crop_size
		# 	)
		# 	inputs = uniform_crop(inputs, crop_size, 1)
		# else:
		# 	raise Exception('the dataset doesn\'t exist')
		# inputs = inputs.float().permute(0, 2, 1, 3, 4)
		slow_pathway = torch.index_select(
				inputs,
				2,
				slow_idx,
			)

		# batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
		# inputs = torch.from_numpy(batch_img)
		# inputs = inputs.float().permute(0, 4, 1, 2, 3)
		if use_gpu:
			if phase == 'train':
				inputs, targets = [Variable(slow_pathway.to(device)),Variable(inputs.to(device))], \
							Variable(targets.to(device))
			if phase == 'val' or phase == 'test':
				inputs, targets = [Variable(slow_pathway.to(device), volatile=True),Variable(inputs.to(device), volatile=True)],\
							 Variable(targets.to(device))
		else:
			if phase == 'train':
				inputs, targets = [Variable(slow_pathway), Variable(inputs)], \
								Variable(targets)
			if phase == 'val' or phase == 'test':
				inputs, targets = [Variable(slow_pathway, volatile=True), Variable(inputs, volatile=True)], \
								(targets)
		if phase == 'train':
			outputs = model(inputs)
		elif phase == 'val' or phase == 'test':
			outputs= None
			with torch.no_grad():
				outputs = model(inputs)
		# if args.every_frame:
		# 	outputs = torch.mean(outputs, 1)
		_, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
		loss = criterion(outputs, targets)
		if phase == 'train':
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step(epoch + batch_idx/len(dset_loaders[phase]))
			warmup_scheduler.dampen()
		# stastics
		running_loss += loss.item() * inputs[0].size(0)
		running_corrects += torch.sum(preds == targets.data)
		running_all += len(inputs[0])
		if batch_idx == 0:
			since = time.time()
		elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
			print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
				running_all,
				len(dset_loaders[phase].dataset),
				100. * batch_idx / (len(dset_loaders[phase])-1),
				running_loss / running_all,
				float(running_corrects) / running_all,
				time.time()-since,
				(time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
	print
	logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
		phase,
		epoch,
		running_loss / len(dset_loaders[phase].dataset),
		float(running_corrects) / len(dset_loaders[phase].dataset))+'\n')
	if phase == 'train':
		torch.save(model.module.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.pt')
		torch.save(optimizer.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.optimizer.pt')
		torch.save(scheduler.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.scheduler.pt')
		torch.save(warmup_scheduler.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.wscheduler.pt')
		torch.save(torch.get_rng_state(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.rng.pt')
		torch.save(torch.cuda.get_rng_state(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.cuda.rng.pt')
		with open(save_path+'/'+args.mode+'_'+str(epoch+1)+'.p', 'wb') as fp:
			pickle.dump(np.random.get_state(), fp, -1)
		return model


def test_adam(args, use_gpu):
	if args.every_frame and args.mode != 'temporalConv':
		save_path = './' + args.mode + '_every_frame'
	elif not args.every_frame and args.mode != 'temporalConv':
		save_path = './' + args.mode + '_last_frame'
	elif args.mode == 'temporalConv':
		save_path = './' + args.mode
	else:
		raise Exception('No model is found!')
	if not args.test and not os.path.isdir(save_path):
		os.mkdir(save_path)
	# logging info
	if args.test:
		filename = './testlog/'+args.mode+'_'+str(args.lr)+'.txt'
	else:
		filename = save_path+'/'+args.mode+'_'+str(args.lr)+'.txt'
	logger_name = "mylog"
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler(filename, mode='a')
	fh.setLevel(logging.INFO)
	logger.addHandler(fh)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	logger.addHandler(console)

	model = lipreading(mode=args.mode, inputDim=256, hiddenDim=512, nClasses=args.nClasses, frameLen=29, every_frame=args.every_frame)
	# reload model
	model = reload_model(model, logger, args.path)
	if torch.cuda.is_available():
		model = model.to(device)
	# define loss function and optimizer
	# TODO: Hamming loss for attribute predictions?
	criterion = SmoothCrossEntropyLoss()

	if torch.cuda.device_count() > 1:
		model = MyDataParallel(model)

	# Batchnorm parameters.
	bn_params = []
	# Non-batchnorm parameters.
	non_bn_parameters = []
	for name, p in model.named_parameters():
		if "bn" in name:
			bn_params.append(p)
		else:
			non_bn_parameters.append(p)
	optim_params = [
		{"params": bn_params, "weight_decay": 0.0},
		{"params": non_bn_parameters, "weight_decay": 1e-4},
	]

	# Memory1 parameters.
	mem_params = []
	# Non-memory parameters.
	non_mem_parameters = []
	for name, p in model.t_encoder1.named_parameters():
		if "memory" in name:
			mem_params.append(p)
		else:
			non_mem_parameters.append(p)
	m1optim_params = [
		{"params": mem_params, "weight_decay": 0.0},
		{"params": non_mem_parameters, "weight_decay": 1e-4},
	]
	# Memory parameters.
	mem_params2 = []
	# Non-memory parameters.
	non_mem_parameters = []
	m2optim_params = [
		{"params": mem_params, "weight_decay": 0.0},
		{"params": non_mem_parameters, "weight_decay": 1e-4},
	]

	if args.mode == 'temporalConv' or args.mode == 'finetuneSelfAttention':
		optimizer = optim.Adam(optim_params, lr=args.lr, betas=(0.9, 0.999))
	elif args.mode == 'backendSelfAttention':
		for param in model.parameters():
			param.requires_grad = False
		for param in model.t_encoder1.parameters():
			param.requires_grad = True
		for param in model.pathway0_backend_conv1.parameters():
			param.requires_grad = True
		for param in model.pathway1_backend_conv1.parameters():
			param.requires_grad = True
		for param in model.backend_conv2.parameters():
			param.requires_grad = True

		# Batchnorm parameters.
		tbn_params = []
		# Non-batchnorm parameters.
		tnon_bn_parameters = []
		for name, p in model.t_encoder1.named_parameters():
			if "bn" in name:
				tbn_params.append(p)
			else:
				tnon_bn_parameters.append(p)
		# Batchnorm parameters.
		bn_params = []
		# Non-batchnorm parameters.
		non_bn_parameters = []
		for name, p in model.pathway0_backend_conv1.named_parameters():
			if "bn" in name:
				bn_params.append(p)
			else:
				non_bn_parameters.append(p)
		for name, p in model.pathway1_backend_conv1.named_parameters():
			if "bn" in name:
				bn_params.append(p)
			else:
				non_bn_parameters.append(p)
		for name, p in model.backend_conv2.named_parameters():
			if "bn" in name:
				bn_params.append(p)
			else:
				non_bn_parameters.append(p)
		optim_params = [
			{"params": bn_params, "weight_decay": 0.0},
			{"params": non_bn_parameters, "weight_decay": 3e-4},
			{"params": tbn_params, "weight_decay": 0.0},
			{"params": tnon_bn_parameters, "weight_decay": 3e-4},
		]
		optimizer = optim.Adam(optim_params, lr=args.lr, betas=(0.9, 0.999))
	else:
		raise Exception('No model is found!')

	dset_loaders, dset_sizes = data_loader(args)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0, last_epoch=-1)
	warmup_scheduler = warmup.LinearWarmup(optimizer, 1000)
	warmup_scheduler.last_step = -1 # initialize the step counter
	#scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=5, half=5, verbose=1)
	#noam_optimizer = NoamOpt(256, 1, 2000, optimizer)
	if args.test:
		train_test(model, dset_loaders, criterion, 0, 'val', optimizer, scheduler, warmup_scheduler, args, logger, use_gpu, save_path)
		train_test(model, dset_loaders, criterion, 0, 'test', optimizer, scheduler, warmup_scheduler, args, logger, use_gpu, save_path)
		return
	if args.resume:
		start_epoch = args.resume_epochs - 1
		optimizer.load_state_dict(torch.load(save_path+'/'+args.mode+'_'+str(start_epoch)+'.optimizer.pt'))
		scheduler.load_state_dict(torch.load(save_path+'/'+args.mode+'_'+str(start_epoch)+'.scheduler.pt'))
		warmup_scheduler.load_state_dict(torch.load(save_path+'/'+args.mode+'_'+str(start_epoch)+'.wscheduler.pt'))
		torch.set_rng_state(torch.load(save_path+'/'+args.mode+'_'+str(start_epoch)+'.rng.pt'))
		torch.cuda.set_rng_state(torch.load(save_path+'/'+args.mode+'_'+str(start_epoch)+'.cuda.rng.pt'))
		with open(save_path+'/'+args.mode+'_'+str(start_epoch)+'.p', 'rb') as fp:
			np.random.set_state(pickle.load(fp))
		# for epoch in range(args.resume_epochs - 1):
		# 	scheduler.step(epoch)
		
	else: 
		start_epoch = 0
	for epoch in range(start_epoch, args.epochs):
		#scheduler.step(epoch)
		model = train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, scheduler, warmup_scheduler, args, logger, use_gpu, save_path)
		train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, scheduler, warmup_scheduler, args, logger, use_gpu, save_path)


def main():
	# Settings
	parser = argparse.ArgumentParser(description='Pytorch Video-only BBC-LRW Example')
	parser.add_argument('--nClasses', default=500, type=int, help='the number of classes')
	parser.add_argument('--path', default='', help='path to model')
	parser.add_argument('--dataset', default='/vulcan/scratch/peratham/lrw/mouth_npz', help='path to dataset')
	parser.add_argument('--mode', default='temporalConv', help='temporalConv, backendSelfAttention, finetuneSelfAttention')
	parser.add_argument('--every-frame', default=False, action='store_true', help='prediction based on every frame')
	parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
	parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size (default: 36)')
	parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', default=30, type=int, help='number of total epochs')
	parser.add_argument('--interval', default=10, type=int, help='display interval')
	parser.add_argument('--test', default=False, action='store_true', help='perform on the test phase')
	parser.add_argument('--resume', default=False, action='store_true', help='resume training on resume epoch')
	parser.add_argument('--resume-epochs', default=20, type=int, help='number of resume epoch')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	use_gpu = torch.cuda.is_available()
	test_adam(args, use_gpu)


if __name__ == '__main__':
	main()
