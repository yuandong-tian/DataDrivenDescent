#ifndef _REGRESSOR_H_
#define _REGRESSOR_H_

#include <algorithm>

#include "image.h"

// Regress (or learn a mapping) from sample to value.
template <typename SrcType, typename DstType>
class Regressor {
public:
  // d_in: input dimension.
  // d_out: output dimension.
  // n: number of samples.
  virtual bool Init(int d_in, int d_out, int n) = 0;

  virtual ImageView<SrcType> GetSampleView(int sample_index) = 0;
  virtual ImageView<DstType> GetParameters(int sample_index) = 0;

  // Set training samples. 
  //virtual bool AddSample(const ImageView<SrcType>& sample, const ImageView<DstType>& value) = 0;

  // Train the model.
  virtual bool Train() = 0;

  // From the sample, get the value.
  virtual bool Retrieve(const ImageView<SrcType>& sample, ImageView<DstType>* value) const = 0;
};

template <typename SrcType, typename DstType>
class NNRegressor : public Regressor<SrcType, DstType> {
private:
  Image<SrcType> samples_;
  Image<DstType> parameters_;
  mutable vector<pair<float, int>> dists_;
  int K_;
  
public:
  NNRegressor() : K_(1) { }

  bool Init(int d_in, int d_out, int n) override {
    samples_ = Image<SrcType>(d_in, n);
    parameters_ = Image<DstType>(d_out, n);
    
    samples_.Zero();
    parameters_.Zero();

    dists_.resize(n);

    return true;
  }

  // Set additional parameters.
  void SetParameters(int K) {
    K_ = K;
  }
  
  // bool AddSample(const ImageView<SrcType>& sample, const ImageView<DstType>& value) override {
  //   // Sanity check.
  //   if (num_samples_ >= samples_.n() || sample.size() != samples_.m() || value.size() != parameters_.m())
  //     return false;
    
  //   // Copy cols.
  //   sample.CopyTo(samples_.col(num_samples_));
  //   value.CopyTo(parameters_.col(num_samples_));
  //   num_samples_++;
    
  //   return true;
  // }

  ImageView<SrcType> GetSampleView(int sample_index) override {
    return samples_.ColView(sample_index);
  }

  ImageView<DstType> GetParameters(int sample_index) override {
    return parameters_.ColView(sample_index);
  }

  // NN does not require training.
  bool Train() override {
    if (samples_.HasNan()) {
      throw std::runtime_error("There is nan in samples.");
    }
    if (parameters_.HasNan()) {
      throw std::runtime_error("There is nan in parameters.");
    }

    return true; 
  }

  bool Retrieve(const ImageView<SrcType>& key, ImageView<DstType>* value) const override {
    // Just L2 distance...
    CHECK_NOTNULL(value);

    if (key.m() != samples_.m() || value->m() != parameters_.m())
      return false;
    
    const SrcType* key_col = key.ptrc();

    for (int i = 0; i < samples_.n(); ++i) {
      float dist = 0.0;
      const SrcType* this_col = samples_.colc(i);
      for (int j = 0; j < samples_.m(); ++j) {
        dist += Distance(this_col[j], key_col[j]);
      }
      dists_[i] = make_pair(dist, i);
    }

    value->Zero();

    // Lazy way to find k nearest-neighbor.
    sort(dists_.begin(), dists_.end());
    const int actual_nn = min(K_, samples_.n());

    LOG_IF(FATAL, actual_nn == 0) << "No samples!" << endl;

    for (int i = 0; i < actual_nn; ++i) {
      const DstType* ptr = parameters_.colc(dists_[i].second);
      for (int j = 0; j < value->m(); ++j) {
        (*value)(j, 0) += ptr[j];
      }
    }
    for (int j = 0; j < value->m(); ++j) {
      (*value)(j, 0) /= actual_nn;
    }

    // D(g_log.PrintInfo("min_index = %d", min_index));
    return true;
  }
};

#endif 
