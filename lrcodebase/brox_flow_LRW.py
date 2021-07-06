import numpy as np
import yaml
import cv2
import os,sys
from PIL import Image
import scipy.misc
import time

import sys
sys.path.append("/cfarhomes/peratham/lrcodebase")
# pyflow
sys.path.append('/cfarhomes/peratham/swpath/pyflow')
import pyflow

from lr_config import *

# from subprocess import Popen, PIPE
from multiprocessing.pool import ThreadPool
import threading

from time import time as timer

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

# optical flow routine
def ExtractFlowImage(vid_urls):
	try:
		im1in, im2in, si, split, vid = vid_urls
		im1path = os.path.join(LRW_frames, si, split, vid, '{:0>6}.png'.format(im1in))
		im2path = os.path.join(LRW_frames, si, split, vid, '{:0>6}.png'.format(im2in))
		im1 = np.array(Image.open(im1path))
		im2 = np.array(Image.open(im2path))
		im1 = im1.astype(float) / 255.
		im2 = im2.astype(float) / 255.
		
		u, v, im2W = pyflow.coarse2fine_flow(
			im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
			nSORIterations, colType)

		flow = np.concatenate((u[..., None], v[..., None]), axis=2)

		if not os.path.exists(os.path.join(LRW_brox, si, split, vid)):
			os.makedirs(os.path.join(LRW_brox, si, split, vid))
		fim1xpath = os.path.join(LRW_brox, si, split, vid,'{:0>6}.jpg'.format(im1in))
		hsv = np.zeros(im1.shape, dtype=np.uint8)
		hsv[:, :, 0] = 255
		hsv[:, :, 1] = 255
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		hsv[..., 0] = ang * 180 / np.pi / 2
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		cv2.imwrite(fim1xpath, rgb)
		return None
	except Exception as e:
		return e

def main(args):
	### <*>body of the program<*> ###
	# get the list of all videos
	num_words_to_process = 25
	n_speaker = 500
	max_sids = 20

	splits = ['train', 'test', 'val']
	srange = list(range(1, max_sids + 1))
	sid = int(args.sidx)
	if sid not in srange:
		raise Exception('sidx out of range.') 
	sids = range((sid - 1) * num_words_to_process, sid * num_words_to_process)

	# for each video directory containing frames
	with open(LRW_class_path,'r') as fp:
		LRW_classes = yaml.safe_load(fp)
	speaker_dirs_sub = [LRW_classes['classes'][s] for s in sids]
	n_threads = 15

	# process each video
	print('Extracting for speaker chunk: {}'.format(sid))
	time.sleep(0.1)
	for si in speaker_dirs_sub:
		for split in splits:
			si_path = os.path.join(LRW_frames, si, split)
			vid_dirs = os.listdir(si_path)
			for vname in vid_dirs:
				vid_urls = [(vf, vf+1, si, split, vname) for vf in range(1, 29)]

				start = timer()
				results = ThreadPool(n_threads).imap_unordered(ExtractFlowImage, vid_urls)
			
				for error in results:
					if error is not None:
						print("error fetching: %s" % (error))
					# if error is None:
					# 	print("%r fetched in %ss" % (url, timer() - start))
					# else:
					# 	print("error fetching %r: %s" % (url, error))
				print("Elapsed Time: %ss" % (timer() - start,))
	return

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Process 25 words at a time.')
	parser.add_argument('--sidx', required=False, type=int, 
			default=1, help='word split to be processed.[1-20]')
	args = parser.parse_args()
	main(args)
