#!/usr/bin/env python
'''
A helper script to extract regions from GRID with dlib..

@author Peratham Wiriyathammabhum
@date Mar 8, 2019
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
import copy

from PIL import Image
import dlib
import cv2
from scipy.misc import imresize

from lr_config import *

# from subprocess import Popen, PIPE
from multiprocessing.pool import ThreadPool
import threading

from time import time as timer

face_predictor_path = os.path.join('/cfarhomes/peratham/swpath/dlib-models','shape_predictor_68_face_landmarks.dat')

# params
# MOUTH_WIDTH = 100
# MOUTH_HEIGHT = 50
# HORIZONTAL_PAD = 0.19
# normalize_ratio = None

def mouth_crop(vtuple):
	im_path, detector, predictor, si, vi, fi = vtuple
	mouth_save_path = os.path.join(GRID_mouths, si, vi, fi)
	if os.path.exists(mouth_save_path):
		return im_path, None
	# params
	MOUTH_WIDTH = 100
	MOUTH_HEIGHT = 50
	HORIZONTAL_PAD = 0.19
	normalize_ratio = None
	img = Image.open(im_path)
	try:
		frame = np.array(img)
		dets = detector(frame, 1)
		shape = None
		for k, d in enumerate(dets):
			shape = predictor(frame, d)
			i = -1
		if shape is None: # Detector doesn't detect face, just return as is
			raise ValueError('Detector doesn\'t detect face.')
		mouth_points = []
		for part in shape.parts():
			i += 1
			if i < 48: # Only take mouth region
				continue
			mouth_points.append((part.x,part.y))
		np_mouth_points = np.array(mouth_points)

		mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

		if normalize_ratio is None:
			mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
			mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

			normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

		new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
		resized_img = imresize(frame, new_img_shape)

		mouth_centroid_norm = mouth_centroid * normalize_ratio

		mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
		mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
		mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
		mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

		mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
		image = Image.fromarray(mouth_crop_image.astype('uint8'), 'RGB')
		mouth_save_path = os.path.join(GRID_mouths, si, vi)
		if not os.path.exists(mouth_save_path):
			os.makedirs(mouth_save_path)
		mouth_save_path = os.path.join(GRID_mouths, si, vi, fi)
		image.save(mouth_save_path)
		return im_path, None
	except Exception as e:
		return im_path, e

def main(args):
	# args parse
	n_speaker = 34
	srange = list(range(1, n_speaker + 1))
	srange.remove(21)
	if args.sidx not in range(len(srange)):
		return
	sid = srange[args.sidx]

	n_threads = 31

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(face_predictor_path)

	# vars
	cannot_detect = []

	# for each video directory containing frames
	speaker_dirs = os.listdir(GRID_frames)
	speaker_dirs_sub = ['s{}'.format(sid)]
	print('Extracting for speaker:{}'.format(speaker_dirs_sub))
	time.sleep(0.1)
	for si in speaker_dirs_sub:
		si_path = os.path.join(GRID_frames, si)
		vid_dirs = os.listdir(si_path)
		for vname in vid_dirs:
			vi_path = os.path.join(GRID_frames, si, vname)
			vframes = os.listdir(vi_path)
			vid_urls = [(os.path.join(vi_path, vf), detector, predictor, si, vname, vf) for vf in vframes]

			with ThreadPool(n_threads) as pool:
				start = timer()
				results = pool.imap_unordered(mouth_crop, vid_urls)
			
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
		default=1, help='speaker to be processed.[1-34] except 21')
	args = parser.parse_args()
	main(args)
