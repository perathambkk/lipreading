#!/usr/bin/env python
'''
Transforms mp4 audio to npz. Code has strong assumptions on the dataset organization!

Based on a script from ICASSP'18 LRW SOTA.
@author Peratham Wiriyathammabhum
@date July 5, 2019
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

from multiprocessing.pool import ThreadPool
import threading

from time import time as timer
import glob
import librosa

def audio_extract(vtuple):
	filename, si, split, vi = vtuple
	audio_save_path = os.path.join(LRW_audios, si, split, vi)
	if os.path.exists(audio_save_path):
		return None
	path_to_save = os.path.join(LRW_audios, si, split)
	if not os.path.exists(path_to_save):
		try:
			os.makedirs(path_to_save)
		except OSError as exc:
			pass
	try:
		data = librosa.load(filename, sr=16000)[0][-19456:]
		path_to_save = os.path.join(path_to_save,'{}.npz'.format(vi))
		np.savez(path_to_save, data=data)
		return None
	except Exception as e:
		return e

def main(args):
	# args parse
	num_words_to_process = 20
	n_speaker = 500
	max_sids = 25

	splits = ['train', 'test', 'val']
	srange = list(range(1, max_sids + 1))
	sid = int(args.sidx)
	if sid not in srange:
		return
	sids = range((sid - 1) * num_words_to_process, sid * num_words_to_process)

	n_threads = 15

	# vars
	cannot_detect = []

	# for each video directory containing frames
	with open(LRW_class_path,'r') as fp:
		LRW_classes = yaml.safe_load(fp)
	speaker_dirs_sub = [LRW_classes['classes'][s] for s in sids]

	print('Extracting for speaker chunk: {}'.format(sid))
	time.sleep(0.1)
	for si in speaker_dirs_sub:
		for split in splits:
			si_path = os.path.join(LRW_videos, si, split)
			vid_dirs = [vd for vd in os.listdir(si_path) if vd.endswith('mp4')]
			vid_urls = [(os.path.join(si_path, vd), si, split, vd) for vd in vid_dirs]

			with ThreadPool(n_threads) as pool:
				start = timer()
				results = pool.imap_unordered(audio_extract, vid_urls)
				
				for error in results:
					if error is not None:
						print("error fetching %r: %s" % (url, error))
				print("Elapsed Time: %ss" % (timer() - start,))

	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Process 20 words at a time.')
	parser.add_argument('--sidx', required=False, type=int, 
		default=1, help='word split to be processed.[1-25]')
	args = parser.parse_args()
	main(args)
