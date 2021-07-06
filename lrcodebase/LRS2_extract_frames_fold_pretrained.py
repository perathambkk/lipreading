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
from time import time as timer

def main():
	n_workers = 4

	allcmds = []

	filepath = os.path.join(LRS2_path, 'pretrain.txt')
	fp  = open(filepath, 'r')
	for vid_dir in fp:
		vid_dir = vid_dir.replace('\n','')
		vsrc = os.path.join(LRS2_pretrained, '{}.mp4'.format(vid_dir))
		vdst = os.path.join(LRS2_frames, 'pretrain', '{}'.format(vid_dir))
		if not os.path.exists(vdst):
			os.makedirs(vdst)
		cmd = ['ffmpeg', '-y', '-i', vsrc, os.path.join(vdst, '%06d.png') ,'-hide_banner','-async', '1', '-r', '25', '-deinterlace'] # '{0:0>6d}.jpg'
		allcmds.append(cmd)
	fp.close()

	print('[Info] Extracting ...')
	start = timer()
	num_batch = math.ceil(len(allcmds)/ float(n_workers))
	for i in range(num_batch):
		start = i * n_workers
		end = (i+1) * n_workers
		if end > len(allcmds):
			end = len(allcmds)
		bcmds = allcmds[start:end]
		procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in bcmds]
		for proc in procs_list:
			proc.communicate()

		print("Elapsed Time: %ss" % (timer() - start,))

	return

if __name__ == '__main__':
	main()
