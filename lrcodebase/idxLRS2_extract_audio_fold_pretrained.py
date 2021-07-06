#!/usr/bin/env python
'''
A helper script to extract frames with ffmpeg..

@author Peratham Wiriyathammabhum
@date Mar 6, 2019
'''
import os, os.path 
import sys
cwd = os.getcwd()
from os.path import expanduser
hp = expanduser("~")

import numpy as np
import yaml, pickle
import pandas as pd
import math
import time

from lr_config import *

from subprocess import Popen, PIPE
from multiprocessing.pool import ThreadPool
import threading

from time import time as timer

def frame_extract(bcmds):
	try:
		procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in bcmds]
		for proc in procs_list:
			proc.communicate()
		return "Done", None
	except Exception as e:
		return bcmds, e

def main(args):
	n_workers = 2
	n_threads = 10

	allcmds = []

	split = 'pretrain.txt'

	filepath = os.path.join(LRS2_path, split)
	fp  = open(filepath, 'r')
	for vid_dir in fp:
		vid_dir = vid_dir.replace('\n','')
		vsrc = os.path.join(LRS2_pretrained, '{}.mp4'.format(vid_dir))
		vdir = vid_dir[:vid_dir.rfind('/')]
		vdst = os.path.join(LRS2_audios, 'pretrain', '{}'.format(vdir))
		if not os.path.exists(vdst):
			os.makedirs(vdst)
		vidname = vid_dir[vid_dir.rfind('/') + 1:]
		cmd = ['ffmpeg', '-y', '-i', vsrc, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-aframes', '25', os.path.join(vdst, '{}.wav'.format(vidname))] # '{0:0>6d}.jpg'
		allcmds.append(cmd)
	fp.close()

	sidx = args.sidx

	sstart = math.ceil((sidx - 1) * len(allcmds) / 3)
	ssend = math.ceil(sidx * len(allcmds) / 3)
	if ssend > len(allcmds):
		ssend = len(allcmds)

	sallcmds = allcmds[sstart:ssend]

	print('[Info] Extracting ...')
	start = timer()
	num_batch = math.ceil(len(sallcmds)/ float(n_workers * n_threads))
	for i in range(num_batch):
		start = i * n_workers * n_threads
		end = (i+1) * n_workers * n_threads
		if end > len(sallcmds):
			end = len(sallcmds)
		bcmds_all = sallcmds[start:end]
		bcmds = []
		j = 0
		while j < len(bcmds_all):
			jend = j + 2
			if jend > len(sallcmds):
				jend = len(sallcmds)
			bcmd = bcmds_all[j:jend]
			bcmds.append(bcmd)
			j += 2

		with ThreadPool(n_threads) as pool:
			start = timer()
			results = pool.imap_unordered(frame_extract, bcmds)
			for url, error in results:
				if error is not None:
					print("error fetching %r: %s" % (url, error))
				# if error is None:
				# 	print("%r fetched in %ss" % (url, timer() - start))
				# else:
				# 	print("error fetching %r: %s" % (url, error))
			print("Elapsed Time: %ss" % (timer() - start,))
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Conduct an experiment with the s2vt model for the youtube videoref data set.')
	parser.add_argument('--sidx', required=False, type=int, 
		default=1, help='split to be processed.[0,1,2]')
	args = parser.parse_args()
	main(args)
