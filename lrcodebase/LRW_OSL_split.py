#!/usr/bin/env python
'''
Transforms mp4 audio to npz. Code has strong assumptions on the dataset organization!

Based on a script from ICASSP'18 LRW SOTA.
@author Peratham Wiriyathammabhum
@date July 5, 2019
'''
import os, os.path 
import sys
import yaml
import random
import shutil

cwd = os.getcwd()
from os.path import expanduser
hp = expanduser("~")

sys.path.append("/cfarhomes/peratham/lrcodebase")
from lr_config import *

random.seed(456)

def osl_split():
	# read classes
	with open('LRW_classes.yaml','r') as fp:
		clist = yaml.safe_load(fp)['classes']
	# create random vector
	n_classes = len(clist)
	val_split = 0.6
	test_split = 0.8
	rvec = list(range(n_classes))
	random.shuffle(rvec)
	clist = [clist[i] for i in rvec] # shuffle
	slist = clist[:int(val_split*n_classes)] #split
	vsplist = clist[int(val_split*n_classes):int(test_split*n_classes)]
	tsplist = clist[int(test_split*n_classes):]
	cdict = {}
	cdict['base_set'] = slist # base set
	cdict['test_support_set'] = tsplist # test support set
	cdict['val_support_set'] = vsplist # val support set
	with open('OSLLRW_split.yaml','w') as fp:
		yaml.dump(cdict, fp)
	return

def main():
	# create split
	if not os.path.exists('OSLLRW_split.yaml'):
		print('[Info] Splitting...')
		osl_split()
	# copy files and set up directories (cp -r)
	# with open('OSLLRW_split.yaml','r') as fp:
	# 	cdict = yaml.safe_load(fp)
	# splits = ['train', 'test', 'val']
	# for w in cdict['support_set']:
	# 	# remove all except an instance in the training set
	# 	bpath = os.path.join(OSLLRW_mouth_npz, w, 'train')
	# 	vfiles = [vn for vn in os.listdir(bpath)]
	# 	random.shuffle(vfiles)
	# 	for i in range(len(vfiles) - 1):
	# 		vfile = os.path.join(bpath, vfiles[i])
	# 		os.remove(vfile)

	return

if __name__ == '__main__':
	main()
