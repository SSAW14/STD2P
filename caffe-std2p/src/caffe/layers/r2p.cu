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
__global__ void CountLabel(const int nthreads,
    const Dtype* correlation, const int target,
    const int height, const int width, int* label_set, int* iSegments) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    *iSegments = 0;
    for(int h = 0;h < height;++h)
    {
      for(int w = 0;w < width;++w)
      {
	int label = correlation[target * height * width + h * width + w];
	int k;
	for(k = 0;k < *iSegments;++k)
	{
	  if(label == label_set[k])
	 {
	    break;
	  }
	}
	if(k == *iSegments)
	{
	  label_set[k] = label;
	  *iSegments = *iSegments + 1;
	}
      }
    }
    for(int j = 0; j < *iSegments-1; j++)
    {
      for(int i = 0; i < *iSegments-1-j; i++)
      {
        if(label_set[i] > label_set[i + 1])
        {
          int temp = label_set[i];
          label_set[i] = label_set[i + 1];
          label_set[i + 1] = temp;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void Forward(const int nthreads,
    const Dtype* bottom_data, const Dtype* correlation,
    const int target, const Dtype* label_set, const int iSegments, int maxSegments,
    const int channels, const int height, const int width,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;

    int label = correlation[target * height * width + h * width + w];
    int k;
    for(k = 0;k < iSegments;++k)
    {
      if(label == label_set[k])
      {
	break;
      }
    }
    if (k < iSegments)
    {
      top_data[index] = bottom_data[c*maxSegments + k];
    }
  }
}

template <typename Dtype>
void R2PLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* corr_data = bottom[1]->gpu_data();
  const Dtype* label_set = bottom[3]->gpu_data();
  const int target = static_cast<int>(*(bottom[2]->cpu_data()));
  const int isegment = static_cast<int>(*(bottom[4]->cpu_data()));
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);

  int count;

// pooling
  count = top[0]->count();
  Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, corr_data, target, label_set, isegment, maxSegments_, channels_, height_, width_, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void Backward(const int nthreads,
    const Dtype* top_diff, const Dtype* correlation,
    const int target, const Dtype* label_set, const int iSegments, const int maxSegments,
    const int channels, const int height, const int width,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % maxSegments;
    if (s >= iSegments)
    {
      return;
    }
    const int c = (index / maxSegments) % channels;
    int label = label_set[s];
    int iNum = 0;
    for(int h = 0;h < height;++h)
    {
      for(int w = 0;w < width;++w)
      {
        if (label == correlation[target * height * width + h * width + w])
	{
	  bottom_diff[index] += top_diff[c*height*width + h*width + w];
	  iNum++;
	}
      }
    }
    bottom_diff[index] = bottom_diff[index]/iNum;
  }
}

template <typename Dtype>
void R2PLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);

  const Dtype* corr_data = bottom[1]->gpu_data();
  const Dtype* label_set = bottom[3]->gpu_data();
  const int target = static_cast<int>(*(bottom[2]->cpu_data()));
  const int isegment = static_cast<int>(*(bottom[4]->cpu_data()));

  int count = channels_*maxSegments_;
  Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, corr_data, target, label_set, isegment, maxSegments_, channels_, height_, width_, bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(R2PLayer);


}  // namespace caffe
