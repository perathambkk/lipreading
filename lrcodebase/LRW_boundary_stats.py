#!/usr/bin/env python
'''
A helper script to extract regions from LRW with dlib.
Mouth ROIs are fixed based on median mouth region across 29 frames.

@author Peratham Wiriyathammabhum
@date Jan 10, 2020
'''
import argparse
import os, os.path 
import sys
import glob
import errno
import pickle
import math
import time
import copy
from multiprocessing import Pool
from time import time as timer

import numpy as np
import cv2
import yaml

cwd = os.getcwd()
from os.path import expanduser
hp = expanduser("~")
sys.path.insert(0, '/cfarhomes/peratham/lrcodebase')
from lr_config import *

from collections import ChainMap
import re

def get_stats(filename):
	stat_dict = {}
	vidname = filename.replace('.txt', '.mp4')

	# .... ex. 'Duration: 0.53 seconds' -> 0.53 float
	# stat_dict['duration'] = ''
	lastline = ''
	with open(filename,'r') as fp:
		lastline = list(fp)[-1]
	x = re.match('\w+: (\d+\.\d+) \w+', lastline)
	duration = float(x.group(1))
	stat_dict['duration'] = duration

	# ....
	# stat_dict['fps'] = ''
	cap = cv2.VideoCapture(vidname)
	fps = cap.get(cv2.CAP_PROP_FPS)
	stat_dict['fps'] = fps
	
	# ....
	# stat_dict['num_frames'] = ''
	stat_dict['num_frames'] = int(round(fps*duration))

	return {filename:stat_dict}

def process_boundary_stats(sample_paths, pool):
	try:
		batch_stats = pool.map(get_stats, sample_paths)
	except:
		print('[Error] {}'.format(file_paths[i]))
	return dict(ChainMap(*batch_stats))

def main(args):
	image_dir = args.dataset
	nthreads = int(args.nthreads)
	split = args.split
	filenames = glob.glob(os.path.join(image_dir, '*', '{}'.format(split), '*.txt'))
	filenames = sorted(filenames)
	total_size = len(filenames)
	pickle.dump( filenames, open( os.path.join(args.outdir, "lrw.{}.filenames.p".format(split)), "wb" ) )

	# ....
	res_dict = {} # result dict {filename:{duration:float.sec, fps:int1, num_frames:int2}}
	current_iter = 0
	chunk = 4*nthreads
	while current_iter < total_size:
		curr_batch_size = chunk if current_iter + chunk <= total_size else total_size - current_iter
		with Pool(nthreads) as pool:
			sample_paths = filenames[current_iter:current_iter+curr_batch_size]
			bdict = process_boundary_stats(sample_paths, pool)
			res_dict = {**res_dict, **bdict}
		current_iter += curr_batch_size
		if current_iter // chunk % 20 == 0:
			print('[Info] Operating...{}'.format(current_iter))
	# ....

	with open(args.outpickle,'wb') as fp:
		pickle.dump(res_dict, fp)

	with open(args.outfile,'w') as fp:
		yaml.dump(res_dict, fp)

	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pytorch Video-only BBC-LRW Example')
	parser.add_argument('--dataset', default='/cfarhomes/peratham/datapath/lrw/lipread_mp4', 
		help='path to dataset')
	parser.add_argument('--split', default='train', 
		help='train, val, test')
	parser.add_argument('--outdir', default='/cfarhomes/peratham/datapath/lrw/boundary_stats/', 
		help='path to output files')
	parser.add_argument('--outfile', default='/cfarhomes/peratham/datapath/lrw/boundary_stats/boundary_stats.yaml', 
		help='path to output yaml')
	parser.add_argument('--outpickle', default='/cfarhomes/peratham/datapath/lrw/boundary_stats/boundary_stats.p', 
		help='path to output pickle')
	parser.add_argument('--nthreads', required=False, type=int, 
		default=64, help='num threads')
	args = parser.parse_args()
	main(args)
