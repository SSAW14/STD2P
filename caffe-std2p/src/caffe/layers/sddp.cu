#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
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
__global__ void CountMaxMin(const int nthreads,
    const Dtype* bottom_data, const Dtype* correlation, const Dtype* label_set,
    const Dtype iSegments, const int maxSegments,
    const int channel, const int height, const int width,
    const float ratio, Dtype* maxvalue, Dtype* minvalue, Dtype* threshold) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % maxSegments;
    if (iSegments <= s)
    {
      return;
    }    
    const int c = (index / maxSegments) % channel;
    const int n = index / maxSegments / channel;
    
    int current_label = label_set[s];

    for(int h = 0;h < height;++h)
    {
      for(int w = 0;w < width;++w)
      {
	if (correlation[n * height * width + h * width + w] == current_label)
	{
	  maxvalue[index] = max(maxvalue[index], bottom_data[n * channel * height * width + c * height * width + h * width + w]);
	  minvalue[index] = min(minvalue[index], bottom_data[n * channel * height * width + c * height * width + h * width + w]);
	}
      }
    }
    threshold[index] = minvalue[index] + (maxvalue[index]-minvalue[index])*ratio - OFFSET;
  }
}

template <typename Dtype>
__global__ void TopKPooling(const int nthreads,
    const Dtype* bottom_data, const Dtype* correlation, const Dtype* label_set, 
    const int iSegments, const int maxSegments,
    const int channel, const int height, const int width,
    const Dtype* threshold, Dtype* top_data, Dtype* top_number_data, const int top_size, double* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % maxSegments;
    if (iSegments <= s)
    {
      return;
    }    
    const int c = (index / maxSegments) % channel;
    const int n = index / maxSegments / channel;    

    int current_label = label_set[s];
    int iNumber = 0;

    for(int h = 0;h < height;++h)
    {
      for(int w = 0;w < width;++w)
      {
        int label = correlation[n * height * width + h * width + w];
        Dtype current_value = bottom_data[n * channel * height * width + c * height * width + h * width + w];

        if (label == current_label && current_value >= threshold[index]) // find the point
        {
          mask[n * channel * height * width + c * height * width + h * width + w] = 1;
          iNumber++;
          top_data[index] += current_value;
        }
      }
    }
    if (iNumber != 0)
    {
      top_data[index] = top_data[index] / iNumber;
      if (top_size > 1)
      {
        top_number_data[index] = iNumber;
      }
    }
  }
}

template <typename Dtype>
void SDDPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
// initialization of label set
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* corr_data = bottom[1]->gpu_data();
  const Dtype* label_set = bottom[3]->gpu_data();
  const int target = static_cast<int>(*(bottom[2]->cpu_data()));
  const int isegment = static_cast<int>(*(bottom[4]->cpu_data()));
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);
  Dtype* top_number_data = NULL;
  if (top.size() > 1)
  {
    top_number_data = top[1]->mutable_gpu_data();
    caffe_gpu_set(top[1]->count(), Dtype(0.), top_number_data);
  }

  const int nSample = bottom[0]->num();

  int count;

  double* idx = blob_idx_.mutable_gpu_data();
  caffe_gpu_set(blob_idx_.count(), double(-1), idx);


  vector<int> vecShape;
  Blob<Dtype> blob_threshold;
  vecShape.assign(1,nSample*maxSegments_*channels_);
  blob_threshold.Reshape(vecShape);
  Dtype *threshold = blob_threshold.mutable_gpu_data();
  caffe_gpu_set(nSample*maxSegments_*channels_, Dtype(-INF), threshold);

  count = nSample*maxSegments_*channels_;

  TopKPooling<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, corr_data, label_set, isegment, maxSegments_, channels_, height_, width_, threshold, top_data, top_number_data, top.size(), idx);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TopKPoolingBackward(const int nthreads,
    const Dtype* top_diff, const Dtype* correlation, const Dtype* label_set,
    const int iSegments, const int maxSegments,
    const int channel, const int height, const int width,
    Dtype* bottom_diff, const double* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channel;
    const int n = index / width / height / channel;

    const int label = static_cast<int>(correlation[n * height * width + h * width + w]);

    int s;
    for(s = 0;s < iSegments;++s)
    {
      if (label == label_set[s])
      {
        break;
      }
    }
    if (s < iSegments && mask[index] >= 0)  //find the point
    {
      bottom_diff[index] = mask[index]*top_diff[n*channel*maxSegments + c*maxSegments + s];
    }
  }
}

template <typename Dtype>
void SDDPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* corr_data = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  
  const Dtype* label_set = bottom[3]->gpu_data();
  const int isegment = static_cast<int>(*(bottom[4]->cpu_data()));

  const double* idx = blob_idx_.gpu_data();

  int count = bottom[0]->count();
  TopKPoolingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, corr_data, label_set, isegment, maxSegments_, channels_, height_, width_, bottom_diff, idx);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(SDDPLayer);


}  // namespace caffe
