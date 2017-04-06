from __future__ import division

import numpy as np

caffe_root = '../caffe-std2p/'

import sys
sys.path.insert(0,caffe_root + 'python')

import caffe
import os
import argparse

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_id', default=0)

args = parser.parse_args()
gpu_id = args.gpu_id

# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net for NYUDv2 using RGB+HHA representation
base_weights = './models/fcn-16s-rgbd-nyud2.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(int(gpu_id))

solver = caffe.SGDSolver('./solver.prototxt')

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

for ii in range(10):
    solver.step(1)

