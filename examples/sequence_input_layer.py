#! Data layer for loading pool4 data and region correspondence.

import sys
sys.path.append('../caffe-std2p/python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkls
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy

data_path = '../nyud2_des/'
corr_path = '../region_correspondence/correspondences/'
labels = '../nyud2_gt425/'

test_frames = 11
train_frames = 11
test_buffer = 1
train_buffer = 1

width = 38
height = 30

def processImageCrop(im_info, transformer):
  im_path_rgb = im_info[0]
  im_path_hha = im_info[1]
  im_path_corr = im_info[2]

  data_in_rgb = np.load(im_path_rgb)
  data_in_hha = np.load(im_path_hha)
  processed_image_rgb = data_in_rgb
  processed_image_hha = data_in_hha
  #import ipdb; ipdb.set_trace()
  f = open(im_path_corr, 'r')
  corr = f.readlines()
  f.close()

  for index_, text_ in enumerate(corr):
    corr[index_] = int(text_)

  corr = np.array(corr).reshape(560,425).transpose()
  processed_image_corr = np.array([[0]*560]*425)
  processed_image_corr = corr

  #import ipdb; ipdb.set_trace()
  return (processed_image_rgb, processed_image_hha, processed_image_corr)

class ImageProcessorCrop(object):
  def __init__(self, transformer):
    self.transformer = transformer
  def __call__(self, im_info):
    return processImageCrop(im_info, self.transformer)

class sequenceGeneratorVideo(object):
  def __init__(self, buffer_size, clip_length, num_videos, video_dict, video_order):
    self.buffer_size = buffer_size
    self.clip_length = clip_length
    self.N = self.buffer_size*self.clip_length
    self.num_videos = num_videos
    self.video_dict = video_dict
    self.video_order = video_order
    self.idx = 0

  def __call__(self):
    label_r = []
    im_paths_rgb = []
    im_paths_hha = []
    im_paths_corr = []
    im_paths_mask = []

    if self.idx + self.buffer_size >= self.num_videos:
      idx_list = range(self.idx, self.num_videos)
      idx_list.extend(range(0, self.buffer_size-(self.num_videos-self.idx)))
    else:
      idx_list = range(self.idx, self.idx+self.buffer_size)
    

    for ii in idx_list:
      key = self.video_order[ii]
      label = self.video_dict[key]['label']
      label_r = label

      frames_rgb = []

      start_id = self.video_dict[key]['order'][0]
      end_id = self.video_dict[key]['order'][1]

      frames_id = range(start_id,end_id+1,3)
      random.shuffle(frames_id)
      frames_id = frames_id[0:min(train_frames,len(frames_id))]
      
	
      if not self.video_dict[key]['target_frame'] in frames_id:
	  frames_id = frames_id[0:test_frames-1]
          frames_id.append(self.video_dict[key]['target_frame'])
      
      frames_id.sort()
      #import ipdb; ipdb.set_trace()
      target = frames_id.index(self.video_dict[key]['target_frame'])
      number = self.video_dict[key]['segment_number']
      label_set = self.video_dict[key]['label_set']
      #import ipdb; ipdb.set_trace()
      frames_rgb = []
      frames_hha = []
      frames_corr = [];

      for i in range(len(frames_id)):
        frames_rgb.append(self.video_dict[key]['rgb_frames'] % frames_id[i])
     
      im_paths_rgb.extend(frames_rgb) 

      for i in range(len(frames_id)):
        frames_hha.append(self.video_dict[key]['hha_frames'] % frames_id[i])
     
      im_paths_hha.extend(frames_hha) 

      for i in range(len(frames_id)):
        frames_corr.append(self.video_dict[key]['correlation'] % frames_id[i])
     
      im_paths_corr.extend(frames_corr)   

    im_info = zip(im_paths_rgb, im_paths_hha, im_paths_corr)

    self.idx += self.buffer_size
    if self.idx >= self.num_videos:
      self.idx = self.idx - self.num_videos

    return label_r, label_set, number, target, im_info
  
def advance_batch(result, sequence_generator, image_processor, pool):
    label_r, label_set, number, target, im_info = sequence_generator()
    #import ipdb; ipdb.set_trace()
    tmp = image_processor(im_info[0])
    #import ipdb; ipdb.set_trace()
    result['data'] = pool.map(image_processor, im_info)
    result['label'] = label_r
    result['target'] = target
    result['label_set'] = label_set
    result['segment_number'] = number
    #import ipdb; ipdb.set_trace()

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)

class videoRead(caffe.Layer):

  def initialize(self):
    random.seed(10)

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.video_list, 'r')
    f_lines = f.readlines()
    f.close()

    f = open(self.gt_list, 'r')
    gt = f.readlines()
    f.close()
    
    f = open(self.folder_list, 'r')
    folders_tmp = f.readlines()
    f.close()

    f = open(self.order_list, 'r')
    orders = f.readlines()
    f.close()

    for index_, text_ in enumerate(f_lines):
      f_lines[index_] = int(text_) 
    f_lines = np.array(f_lines)

    for index_, text_ in enumerate(gt):
      gt[index_] = int(text_)  
    gt = np.array(gt)

    target = gt[f_lines-1]
 
    for index_, text_ in enumerate(orders):
      orders[index_] = int(text_) 

    orders = np.array(orders).reshape(len(orders)/2,2)
    orders = orders[f_lines-1,:]

    folders = []
    for index_, text_ in enumerate(f_lines):
      folders.append(folders_tmp[text_-1][0:len(folders_tmp[text_-1])-1])

    video_dict = {}
    current_line = 0
    self.video_order = []
# generating video data
    for ix, line in enumerate(f_lines):
      print line
      video = line
      target_id = target[ix]
      video_dict[video] = {}
      video_dict[video]['folder'] = folders[ix]

      video_dict[video]['target_frame'] = target_id
      video_dict[video]['rgb_frames'] = '%s%s/pool4_' %(self.path_to_data, video_dict[video]['folder']) + '%05d.npy'
      video_dict[video]['hha_frames'] = '%s%s/pool4_hha_' %(self.path_to_data, video_dict[video]['folder']) + '%05d.npy'
      video_dict[video]['correlation'] = '%s' % corr_path + '%04d/'%f_lines[ix] + '%05d.txt'
      video_dict[video]['order'] = orders[ix,:]


      temp = np.array(Image.open('%s%03d.bmp'%(labels,int(video))))
      temp = temp[None,:,:]
      temp = temp - 1

      video_dict[video]['label'] = temp


      f = open(video_dict[video]['correlation'] % target_id, 'r')
      corr = f.readlines()
      f.close()

      for index_, text_ in enumerate(corr):
        corr[index_] = int(text_)

      corr_target = np.array(corr).reshape(560,425).transpose()
      temp1 = np.unique(corr_target)
      temp1 = temp1.reshape([temp1.size])

      number = temp1.size
      temp = -1*np.ones([self.max_segments])
      temp[0:temp1.size] = temp1
      label_set = temp

      video_dict[video]['segment_number'] = number
      video_dict[video]['label_set'] = label_set
      self.video_order.append(video) 

    #import ipdb; ipdb.set_trace()
    self.video_dict = video_dict
    self.num_videos = len(video_dict.keys())
    
    #set up data transformer
    shape = (self.N, self.channels, self.height_out, self.width_out)
        
    self.transformer = caffe.io.Transformer({'pool4': shape, 'pool4_hha': shape})

    self.thread_result = {}
    self.thread = None
    pool_size = 1

    self.image_processor = ImageProcessorCrop(self.transformer)
    self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.frames, self.num_videos, self.video_dict, self.video_order)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, self.pool)
    self.dispatch_worker()
    self.top_names = ['pool4', 'pool4_hha', 'correlation', 'target', 'label_set', 'segment_number', 'label']
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))

    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'pool4':
        shape = (self.N, self.channels, self.height_in, self.width_in)
      elif name == 'pool4_hha':
        shape = (self.N, self.channels, self.height_in, self.width_in)
      elif name == 'correlation':
        shape = (self.N, 1, self.height_out, self.width_out)
      elif name == 'label':
        shape = (1, 1, self.height_out, self.width_out)
      elif name == 'target':
        shape = (1, 1, 1, 1)
      elif name == 'segment_number':
        shape = (1, 1, 1, 1)
      elif name == 'label_set':
        shape = (1, 1, 1, self.max_segments)

      top[top_index].reshape(*shape)


  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    if self.thread is not None:
      self.join_worker() 

    #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
    new_result_data_rgb = [None]*len(self.thread_result['data'])    
    new_result_data_hha = [None]*len(self.thread_result['data'])    
    new_result_data_corr = [None]*len(self.thread_result['data'])    
    new_result_label = self.thread_result['label']    
    new_result_target = self.thread_result['target']   
    new_result_segment_number = self.thread_result['segment_number']    
    new_result_label_set = self.thread_result['label_set']    
    #import ipdb; ipdb.set_trace()
    for i in range(self.frames * self.buffer_size):
      new_result_data_rgb[i] = self.thread_result['data'][i][0]
      new_result_data_hha[i] = self.thread_result['data'][i][1]
      new_result_data_corr[i] = self.thread_result['data'][i][2]

    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'pool4':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data_rgb[i] 
      elif name == 'pool4_hha':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data_hha[i] 
      elif name == 'correlation':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data_corr[i] 
      elif name == 'label':
        top[top_index].data[0, ...] = new_result_label
      elif name == 'target':
        top[top_index].data[0, 0, 0, 0] = new_result_target
      elif name == 'segment_number':
        top[top_index].data[0, 0, 0, 0] = new_result_segment_number
      elif name == 'label_set':
        top[top_index].data[0, 0, 0, ...] = new_result_label_set

    
    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class videoRead(videoRead):
  def initialize(self):
    self.train_or_test = 'train'
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 512
    self.height_in = 39
    self.width_in = 48
    self.height_out = 425
    self.width_out = 560
    self.max_segments = 510
    self.path_to_data = data_path
    self.video_list = '../config/nyud2_trainlist.txt'
    self.gt_list = '../config/nyud2_target.txt'
    self.folder_list = '../config/nyud2_folder.txt'
    self.order_list = '../config/nyud2_order.txt'

