#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define INF 9999999
#define OFFSET 0

using namespace std;

namespace caffe {

template <typename Dtype>
__global__ void CountMaxMin(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_map,
    const int nSample, const int maxSegments, const int channel, const float ratio,
    Dtype* maxvalue, Dtype* minvalue, Dtype* threshold) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % maxSegments;
    const int c = index / maxSegments;

    bool bFound = false;
    for(int n = 0;n < nSample;++n)
    {
      if (bottom_map[n*channel*maxSegments+c*maxSegments+s] > 0) //no matching
      {
        Dtype current_value = bottom_data[n * channel * maxSegments + c * maxSegments + s];
        maxvalue[index] = max(maxvalue[index], current_value);
        minvalue[index] = min(minvalue[index], current_value);
	bFound = true;
      }
    }
    if (bFound)
    {
      threshold[index] = minvalue[index] + (maxvalue[index]-minvalue[index])*ratio - OFFSET;
    }
  } 
}

template <typename Dtype>
__global__ void TopKPooling(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_map,
    const int nSample, const int maxSegments, const int channel,
    const Dtype* threshold, Dtype* top_data, double* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % maxSegments;
    const int c = index / maxSegments;

    int iFrames = 0;
    for(int n = 0;n < nSample;++n)
    {
      if (bottom_map[n*channel*maxSegments+c*maxSegments+s] > 0 && bottom_data[n*channel*maxSegments+c*maxSegments+s] >= threshold[index])
      {
	top_data[index] += bottom_data[n*channel*maxSegments+c*maxSegments+s];
        mask[n*channel*maxSegments+c*maxSegments+s] = 1;
        iFrames++;
      }
    }
    top_data[index] = top_data[index]/iFrames;
  }
}

template <typename Dtype>
void TDDPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_map = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);

  double* idx = blob_idx_.mutable_gpu_data();
  caffe_gpu_set(blob_idx_.count(), double(-1), idx);

  const int nSample = bottom[0]->num();

  int count;
// count max, min and threshod
  vector<int> vecShape;
  Blob<Dtype> blob_threshold;
  vecShape.assign(1,maxSegments_*channels_);
  blob_threshold.Reshape(vecShape);
  Dtype *threshold = blob_threshold.mutable_gpu_data();
  caffe_gpu_set(maxSegments_*channels_, Dtype(-INF), threshold);

  count = maxSegments_*channels_;

// pooling
  TopKPooling<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, bottom_map, nSample, maxSegments_, channels_, threshold, top_data, idx);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TopKPoolingBackward(const int nthreads,
    const Dtype* top_diff, const int maxSegments, const int channel,
    Dtype* bottom_diff, const double* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % maxSegments;
    const int c = (index / maxSegments) % channel;

    if (mask[index] >= 0)
    {
      bottom_diff[index] = mask[index]*top_diff[c*maxSegments + s];
    }
  }
}

template <typename Dtype>
void TDDPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  
  const double* idx = blob_idx_.gpu_data();

  int count = bottom[0]->count();
  TopKPoolingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, maxSegments_, channels_, bottom_diff, idx);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(TDDPLayer);


}  // namespace caffe
