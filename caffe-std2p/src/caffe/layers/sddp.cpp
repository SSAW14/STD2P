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
void SDDPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num() == bottom[1]->num())
    << "Mask and data should be the same.";

// default
  ratio_ = 0.5;
  maxSegments_ = 300;

  SDDPParameter sddp_param = this->layer_param_.sddp_param();

  maxSegments_ = sddp_param.max_segments();
  ratio_ = sddp_param.ratio();

  CHECK(ratio_ >= 0 && ratio_ <= 1)
      << "Pixel number for fusion should be in the set [0,1]";

  CHECK(maxSegments_ > 0)
    << "Max segments number should be positive";
}

template <typename Dtype>
void SDDPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(bottom[0]->num(), channels_, maxSegments_, 1);
  if (top.size() > 1)
  {
    top[1]->Reshape(bottom[0]->num(), channels_, maxSegments_, 1);
  }
  blob_idx_.Reshape(bottom[0]->num(),channels_,height_,width_);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void SDDPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED; 
}

template <typename Dtype>
void SDDPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(SDDPLayer);
#endif

INSTANTIATE_CLASS(SDDPLayer);
REGISTER_LAYER_CLASS(SDDP);

}  // namespace caffe
