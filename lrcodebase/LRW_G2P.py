import numpy as np
import yaml
import os,sys
import time

import sys
sys.path.append("/cfarhomes/peratham/lrcodebase")
from g2p_en import G2p

from lr_config import *

def main(args):
	with open(args.classpath, 'r') as fp:
		clist = yaml.safe_load(fp)['classes']
	g2p = G2p()
	pdict = {}
	for i, c in enumerate(clist):
		pdict[c] = g2p(c)

	with open(args.phonemepath, 'w') as fp:
		yaml.dump(pdict, fp)
	return

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Process 25 words at a time.')
	parser.add_argument('--classpath', required=False, type=str, 
			default='/cfarhomes/peratham/lrcodebase/LRW_classes.yaml', help='path to LRW classpath.')
	parser.add_argument('--phonemepath', required=False, type=str, 
			default='/cfarhomes/peratham/lrcodebase/LRW_phonemes.yaml', help='path to LRW phonemepath.')
	args = parser.parse_args()
	main(args)
