#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <stdlib.h>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define INF 9999999999

using namespace std;

namespace caffe {

using std::min;
using std::max;


template <typename Dtype>
void R2PLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  maxSegments_ = bottom[0]->height();
  height_ = bottom[1]->height();
  width_ = bottom[1]->width();
  top[0]->Reshape(1, channels_, height_, width_);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void R2PLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED; 
}

template <typename Dtype>
void R2PLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(R2PLayer);
#endif

INSTANTIATE_CLASS(R2PLayer);
REGISTER_LAYER_CLASS(R2P);

}  // namespace caffe
