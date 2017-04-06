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
void TDDPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num() == bottom[1]->num())
    << "Mask and data should be the same.";

  ratio_ = 0.5;

  TDDPParameter tddp_param = this->layer_param_.tddp_param();
  ratio_ = tddp_param.ratio();

  CHECK(ratio_ >= 0 && ratio_ <= 1)
      << "Pixel number for fusion should be in the set [0,1]";
}

template <typename Dtype>
void TDDPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  maxSegments_ = bottom[0]->height();
  top[0]->Reshape(1, channels_, maxSegments_, 1);
  blob_idx_.Reshape(bottom[0]->num(),channels_,maxSegments_,1);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void TDDPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED; 
}

template <typename Dtype>
void TDDPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(TDDPLayer);
#endif

INSTANTIATE_CLASS(TDDPLayer);
REGISTER_LAYER_CLASS(TDDP);

}  // namespace caffe
