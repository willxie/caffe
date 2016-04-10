#ifndef CAFFE_ATARI_LAYERS_HPP_
#define CAFFE_ATARI_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

namespace caffe {


template <typename T>
class Net;

template <typename Dtype>
class AtariDataLayer : public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit AtariDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~AtariDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

   // The thread's function
  virtual void InternalThreadEntry();

 protected:

  struct BatchIdx {
    int dir_;
    int img_;

    BatchIdx(int dir, int img) 
      : dir_(dir), img_(img) {
    }

    BatchIdx(const BatchIdx& rhs) {
      this->dir_ = rhs.dir_;
      this->img_ = rhs.img_;
    }
  };

  void LoadData();
  void SampleBatch();

  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_act_;
  Blob<Dtype> prefetch_clip_;

  bool output_act_;
  bool output_clip_;
  bool streaming_;
  bool load_to_mem_;

  int act_idx_;
  int clip_idx_;
  int batch_size_, channels_, height_, width_, size_;
  int num_act_;
  int num_frame_;

  Blob<Dtype> datum_mean_blob_;
  const Dtype* datum_mean_;

  std::vector<std::vector<int> > acts_;
  std::vector<std::vector<cv::Mat> > imgs_;
  std::vector<BatchIdx> batch_idx_;
  std::vector<BatchIdx> sample_batch_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
