#ifndef _DATA_DRIVEN_DESCENT_H_
#define _DATA_DRIVEN_DESCENT_H_

#include "regressor.h"
#include "deformations.h"
#include "data_driven_descent.pb.h"

typedef float FloatType;

template <typename T>
T L2NormSqr(const std::vector<T> &v) {
  T res = 0;
  for (int i = 0; i < (int)v.size(); i ++) {
    res += v[i] * v[i];
  }
  return res;
};

template <typename T>
class Representer {
public:
  // Compute Representation at layer t.
  bool ComputeRepresentation(const ImageView<T>& img) {
    identity_view_ = &img;

    Resize(img, &image_pyramid_[0]);
    for (int t = 1; t < num_layer_; ++t) {
      Resize(image_pyramid_[t - 1], &image_pyramid_[t]);
    }

    return true;
  }
      
  const ImageView<T>& GetRepresentation(int t) const {
    // return *identity_view_;
    return image_pyramid_[t];
  }

private:
  const ImageView<T>* identity_view_;

  // Image pyramid.
  vector<Image<T>> image_pyramid_;
  //
  int num_layer_;
};

// Generate parameters from specification.
template <typename ParamType>
class FieldServer {
private:
  ParametricField<ParamType>* field_;
  // Use when ONLY_TRANSLATION is specified.
  std::unique_ptr<TranslationalField<ParamType>> translation_field_;
  Image<ParamType> parameters_;

  bool is_only_translational_;

public:
  FieldServer(ParametricField<ParamType>* field)
      : field_(field), 
        parameters_(field->dof(), 1), 
        is_only_translational_(false) {
    translation_field_.reset(new TranslationalField<ParamType>(field_->m(), field_->n()));
  }

  bool RandomParameters(const ddd::SampleSpec& sample_spec) {
    // Generate random magnitude
    if (sample_spec.sample_type() == ddd::ONLY_TRANSLATION) {
      ParamType rx = (2 * uniformRand() - 1) * sample_spec.sigma();
      ParamType ry = (2 * uniformRand() - 1) * sample_spec.sigma();

      int i = 0;
      for (; i < parameters_.m() / 2; ++i) {
        parameters_(i, 0) = rx;
      }
      for (; i < parameters_.m(); ++i) {
        parameters_(i, 0) = ry;
      }      

      translation_field_->SetP({rx, ry});
      is_only_translational_ = true;
    } else {
      const float r = pow(uniformRand(), sample_spec.power()) * sample_spec.sigma();
      // Generate a random direction.
      float mag = 0.0;
      for (int j = 0; j < parameters_.m(); ++j) {
        const float v = gaussianRand();
        parameters_(j, 0) = v * r;
        mag += v * v;
      }
      mag = sqrt(mag);
      for (int j = 0; j < parameters_.m(); ++j) {
        parameters_(j, 0) /= mag;
      } 

      field_->SetP(parameters_);
      is_only_translational_ = false;
    }
    return true;
  }

  void ClearTranslation() { is_only_translational_ = false; }

  // const ImageView<ParamType>& GetParameters() const { return parameters_; }
  ParametricField<ParamType>* GetField() const {
    return is_only_translational_ ? translation_field_.get() : field_;
  }

  const ImageView<ParamType>& GetParameters() const { return parameters_; }
};

// Sample the pixels from images. You can either sample densely or sample sparsely.
template <typename ImgType, typename ParamType>
class SampleGenerator {
private:
  FieldServer<ParamType>* field_;

public:
  SampleGenerator(FieldServer<ParamType>* field) : field_(field) { }

  bool Generate(const ImageView<ImgType>& template_img, ddd::WarpType warp_type, 
    ImageView<ImgType>* deform) {
    CHECK_NOTNULL(deform);

    if (field_ == nullptr) return false;
    if (template_img.m() != deform->m() || template_img.n() != deform->n()) return false;

    if (warp_type == ddd::BACKWARD) {
      return WarpBackward(template_img, *field_->GetField(), deform);
    } else if (warp_type == ddd::FORWARD) {
      return WarpForward(template_img, *field_->GetField(), deform);
    } else return false;
  }

  bool Generate(const ImageView<ImgType>& template_img, ddd::WarpType warp_type, const vector<Point>& ps,
    ImageView<ImgType>* result) {
    CHECK_NOTNULL(result);
    CHECK_EQ(ps.size(), result->m());

    if (field_ == nullptr) return false;
    if (warp_type == ddd::BACKWARD) {
      return WarpBackward(template_img, *field_->GetField(), ps, result);
    } else {
      return false;
    }
  }

  FieldServer<ParamType>* GetFieldServer() { return field_; }
};

//  Image<ImgType> generated_image_;

// Define an image region and its associated parameter subset.
template <typename T1, typename T2>
struct Predictor {
  ddd::Region region;

  vector<int> sample_indices;
  std::unique_ptr<NNRegressor<T1, T2>> regressor;

  Predictor() { }
  Predictor(const Predictor& predictor) { 
    region = predictor.region;
    regressor.reset(nullptr);
  }  

  string PrintInfo() const {
    return StringPrintf("#sample_indices = %d, region = [%d %d %d %d]", 
      sample_indices.size(), region.left(), region.top(), region.width(), region.height());
  }

  ImageView<T1> CropRegion(const ImageView<T1>& image) {
    return image.ViewC(region.left(), region.top(), region.width(), region.height());
  }

  void SampleView(const ImageView<T1>& sampled_pixels, ImageView<T1>* subset) {
    CHECK_EQ(subset->rows(), sample_indices.size());

    const T1* input_ptr = sampled_pixels.ptrc();
    T1* output_ptr = subset->ptr();

    for (const int& i : sample_indices) {
      *output_ptr++ = input_ptr[i];
    }

    // return sampled_pixels.RowViewC(start_index, end_index - start_index);
  }

  void CropSubset(const ImageView<T2>& parameters, ImageView<T2>* subset) {
    T2* ptr = subset->ptr();    
    for (int i = 0; i < region.subsets_size(); ++i) {
      *ptr++ = parameters(region.subsets(i), 0);
    }
    //return parameters.RowView(0, subset.size());
  }

  void CropSubset(const vector<T2>& parameters, ImageView<T2>* subset) {
    CHECK_EQ(region.subsets_size(), subset->rows());
    T2* ptr = subset->ptr();
    for (int i = 0; i < region.subsets_size(); ++i) {
      *ptr++ = parameters[region.subsets(i)];
    }
    //return parameters.RowView(0, subset.size());
  }

  bool CheckParamCondition(const ImageView<T2>& parameters) {
    T2 max_value = 0.0;
    for (int i = 0; i < region.subsets_size(); ++i) {
      const T2& value = parameters(region.subsets(i), 0);
      T2 abs_val = value >= 0.0 ? value : -value;
      max_value = max(max_value, abs_val);
    }
    return max_value >= region.min_magnitude() && 
        max_value <= region.max_magnitude();
  }

  // void AddSamplePoints(int nSide, vector<Point>* samples) {
  //   CHECK_NOTNULL(samples);

  //   start_index = samples->size();

  //   for (int i = 0; i < nSide; ++i) {
  //     const int x = region.left() + i * region.width() / nSide;
  //     for (int j = 0; j < nSide; ++j) {
  //       const int y = region.top() + j * region.height() / nSide;
  //       samples->push_back(Point(x, y));
  //     }
  //   }

  //   end_index = samples->size();
  // }

  void FilterLocs(const vector<Point>& locs) {
    const int right = region.left() + region.width();
    const int bottom = region.top() + region.height();

    sample_indices.clear();
    for (int i = 0; i < locs.size(); ++i) {
      const int x = locs[i].x;
      const int y = locs[i].y;
      if (x >= region.left() && y >= region.top() && 
          x < right && y < bottom) {
        sample_indices.push_back(i);
      }
    }
  }  
};

int GetRegionArea(const ddd::Region& region) {
  return region.width() * region.height();
}

template <typename ImgType, typename ParamType>
class RegressorEmsemble {
private:
  vector<vector<Predictor<ImgType, ParamType>>> predictors_;

  // Pixel samples in each layer.
  vector<vector<Point>> sample_locs_;

  // Temporary information.
  vector<Image<ImgType>> sample_imgs_;

  int d_output_;

  void SetupPredictors(const ddd::AlgSpec& alg_spec) {
    // Initialize the predictors.
    predictors_.resize(alg_spec.layers_size());
    sample_locs_.resize(alg_spec.layers_size());
    sample_imgs_.resize(alg_spec.layers_size());

    for (int t = 0; t < alg_spec.layers_size(); ++t) {
      LOG(INFO) << "Setup predictors, layer = " << t << endl;
      const ddd::LayerSpec& layer_spec = alg_spec.layers(t);
      vector<Predictor<ImgType, ParamType>>& pt = predictors_[t];

      // Get union of regions.
      int xmax = 0, ymax = 0, xmin = 10000, ymin = 10000;
      for (int i = 0; i < layer_spec.regions_size(); ++i) {
        const ddd::Region& region = layer_spec.regions(i);

        xmin = min(xmin, region.left());
        ymin = min(ymin, region.top());
        xmax = max(xmax, region.left() + region.width());
        ymax = max(ymax, region.top() + region.height());
      }

      // Get sample points from the image.
      const int nSide = layer_spec.num_samples_per_dim();
      for (int i = 0; i < nSide; ++i) {
        const int x = xmin + i * (xmax - xmin) / nSide;
        for (int j = 0; j < nSide; ++j) {
          const int y = ymin + j * (ymax - ymin) / nSide;
          sample_locs_[t].push_back(Point(x, y));
        }
      }

      pt.resize(layer_spec.regions_size());
      for (int i = 0; i < layer_spec.regions_size(); ++i) {
        const ddd::Region& region = layer_spec.regions(i);
        pt[i].region = region;
        pt[i].regressor.reset(new NNRegressor<ImgType, ParamType>());

        // Set up the samples
        pt[i].FilterLocs(sample_locs_[t]);

        // pt[i].AddSamplePoints(layer_spec.regions(i).num_samples_per_dim(), &sample_locs_[t]);

        pt[i].regressor->Init(pt[i].sample_indices.size(), region.subsets_size(), layer_spec.sample_spec().num_samples());
        pt[i].regressor->SetParameters(alg_spec.nearest_neighbor());
      }

      // 
      sample_imgs_[t].ReAllocate(sample_locs_[t].size(), 1);
    }
  }

  void SampleOnImage(const ImageView<ImgType>& img, const vector<Point>& samples, ImageView<ImgType>* result) {
    CHECK_NOTNULL(result);
    ImgType* ptr = result->ptr();
    for (const Point& p : samples) {
      *ptr++ = img(p.x, p.y);
    }
  }

public:
  RegressorEmsemble() : d_output_(0) { }

  // Generate data.
  bool GenerateAndTrain(const ImageView<ImgType>& template_img,
    Representer<ImgType>* representer,
    SampleGenerator<ImgType, ParamType>* generator,
    const ddd::DeformationSpec& def_spec, const ddd::AlgSpec& alg_spec) {
    if (representer == nullptr || generator == nullptr) return false;

    FieldServer<ParamType>* field_server = generator->GetFieldServer();
    d_output_ = field_server->GetField()->dof();
    SetupPredictors(alg_spec);

    Image<ImgType> generated_img(template_img.m(), template_img.n());

    for (int t = 0; t < alg_spec.layers_size(); ++t) {
      const ddd::LayerSpec& layer_spec = alg_spec.layers(t);
      LOG(INFO) << "Generate samples, layer = " << t << " #Sample = " 
                << layer_spec.sample_spec().num_samples() << endl;

      // Generate samples.
      for (int i = 0; i < layer_spec.sample_spec().num_samples(); ++i) {
        // Generate random deformation.
        field_server->RandomParameters(layer_spec.sample_spec());
        generator->Generate(template_img, def_spec.warp_type(), &generated_img);
        representer->ComputeRepresentation(generated_img);
        const ImageView<ImgType>& rep = representer->GetRepresentation(t);

        // Sample.
        SampleOnImage(rep, sample_locs_[t], &sample_imgs_[t]);

        // Save.
        for (Predictor<ImgType, ParamType>& p : predictors_[t]) {
          ImageView<ImgType> sample_img = p.regressor->GetSampleView(i);
          p.SampleView(sample_imgs_[t], &sample_img);
          ImageView<ParamType> sample_p = p.regressor->GetParameters(i);
          p.CropSubset(field_server->GetParameters(), &sample_p);
        }
      }

      // Train all the regressors.
      for (Predictor<ImgType, ParamType>& p : predictors_[t]) {
        p.regressor->Train();
        // const ddd::Region& region = p.region;
        // cout << StringPrintf("Box: layer: %d: Size: [%d %d %d %d]", region.layer(), region.left(), region.top(), region.width(), region.height()) << endl;
        // cout << "Number of samples = " << p.regressor->GetNumOfSamples() << endl;
        // cout << "Subset: ";
        // // Check subset size.
        // for (const int& s : region.subsets()) {
        //   if (s < 0 || s >= parameters_.size()) return false;
        //   cout << s << ",";
        // }
        // cout << endl;
      }
    }
    
    return true;
  }

  bool Predict(const Representer<ImgType>& representer, 
    ddd::WarpType warp_type, SampleGenerator<ImgType, ParamType>* generator, 
    int t, Image<ParamType>* parameters) {
    if (parameters == nullptr) return false;
    if (parameters->size() != d_output_) return false;

    const ImageView<ImgType>& rep = representer.GetRepresentation(t);

    // Do a pull-back sampling.
    ddd::WarpType pullback_type = (warp_type == ddd::FORWARD ? ddd::BACKWARD : ddd::FORWARD); 
    // g_log.StartTiming();
    if (!generator->Generate(rep, pullback_type, sample_locs_[t], &sample_imgs_[t])) return false;

    parameters->Zero();

    // cout << "RegressorEmsemble::Predict" << " Output dimension = " << parameters_.size() << endl;

    vector<int> counter(d_output_, 0);
    Image<ImgType> img_buf(sample_locs_[t].size(), 1);
    Image<ParamType> parameter_buf(d_output_, 1);

    for (Predictor<ImgType, ParamType>& p : predictors_[t]) {
      const ddd::Region& region = p.region;

      ImageView<ParamType> param_view = parameter_buf.RowView(0, region.subsets_size());
      ImageView<ImgType> img_view = img_buf.RowView(0, p.sample_indices.size());      

      p.SampleView(sample_imgs_[t], &img_view);

      if (p.regressor->Retrieve(img_view, &param_view)) {
        LOG_IF(FATAL, param_view.HasNan()) << "region prediction has nan: " << p.PrintInfo() << endl;

        for (int i = 0; i < region.subsets_size(); ++i) {
          const int index = region.subsets(i);
          counter[index]++;
          (*parameters)(index, 0) += param_view(i, 0);
        }
      }
    }

    LOG_IF(FATAL, parameters->HasNan()) << "Before averaging, estimated parameters has nan." << endl;

    // Finally compute the average.
    bool all_counter_zero = true;
    for (int i = 0; i < parameters->size(); ++i) {
      if (counter[i] > 0) {
        (*parameters)(i, 0) /= counter[i];
        all_counter_zero = false;
      }
    }

    LOG_IF(FATAL, parameters->HasNan()) << "Estimated parameters has nan." << endl;
    LOG_IF(INFO, all_counter_zero) << "RegressorEmsemble::Predict(): All counter is zero!!" << endl;

    return true;
  }

  int GetOutputDim() const { return d_output_; }
  int GetNumOfLayers() const { return (int)predictors_.size(); }
};

template <typename T1, typename T2>
class DataDrivenDescent {
public:
  DataDrivenDescent(RegressorEmsemble<T1, T2>* regressor_emsemble, 
                    const ddd::AlgSpec& alg_spec)
        : regressor_emsemble_(regressor_emsemble), 
          delta_parameters_(regressor_emsemble->GetOutputDim(), 1), 
          alg_spec_(alg_spec) {
  }

  // result can be nullptr, if so, do not return intermediate results.
  bool Estimate(const ImageView<T1>& deformed_img, 
                Representer<T1>* representer, SampleGenerator<T1, T2>* generator, 
                ddd::WarpType warp_type, ddd::Result* result) {
    if (regressor_emsemble_ == nullptr || representer == nullptr) return false;

    FieldServer<T2>* field_server = generator->GetFieldServer();

    field_server->ClearTranslation();
    ParametricField<T2>* field = field_server->GetField();
    field->Zero();

    representer->ComputeRepresentation(deformed_img);

    LOG(INFO) << "Dimension of parameter: " << field->dof() << endl;

    for (int t = 0; t < regressor_emsemble_->GetNumOfLayers(); ++t) {
      LOG(INFO) << "Iteration " << t << endl;
      for (int j = 0; j < alg_spec_.layers(t).num_iterations(); ++j) {
        if (!regressor_emsemble_->Predict(*representer, warp_type, generator, t, &delta_parameters_)) {
          LOG(FATAL) << "Prediction is wrong!" << endl;
          return false;          
        }

        field->AddP(delta_parameters_);
      }

      // D(g_log.PrintVectorRow(parameters, "Delta Parameter:"););
      if (alg_spec_.dump_intermediate()) SaveCurrentFrame(t, *representer, *field, result);    
    }

    if (result != nullptr) {
      // Put down the final estimate.
      result->clear_estimates();
      for (int i = 0; i < field->dof(); ++i) {
        result->add_estimates(field->GetP()[i]);
      }
    }

    LOG(INFO) << "Data-driven descent finished." << endl;
    
    // double time_spent = g_log.EndTiming();
    
    // g_log.PrintInfo("Data-Driven Descent Done Successfully. #Iteration = %d, Time Spent = %lf", t, time_spent);
    return true;
  }

private:
  // Prediction at layer t, by estimating the parameters within a few rectangles.
  bool SaveCurrentFrame(int t, const Representer<T1>& representer, 
                        const ParametricField<T2>& field, 
                        ddd::Result* result) {
    if (result == nullptr) return false;

    ddd::Result::Frame* frame = result->add_frames();
    frame->set_t(t);

    const ImageView<T1>& rep_view = representer.GetRepresentation(t);
    auto* rep_frame = frame->mutable_representation();
    typedef std::remove_pointer<decltype(rep_frame)>::type RepField;

    int reserve_size = rep_view.size() * sizeof(T1) / sizeof(RepField::value_type);
    rep_frame->Reserve(reserve_size);

    // cout << "Iteration " << t << ": reserve = " << reserve_size << ", actual = " << rep_frame->size() << endl;
    for (int k = 0; k < rep_view.size(); ++k) {
      const T1& val = rep_view.ptrc()[k];
      const RepField::value_type *ptr = reinterpret_cast<const RepField::value_type *>(&val);
      for (int l = 0; l < sizeof(T1) / sizeof(RepField::value_type); ++l) {
        frame->add_representation(ptr[l]);
      }
    }
    // rep_view.CopyTo(reinterpret_cast<T1 *>(rep_frame->mutable_data()));

    frame->clear_estimates();
    for (int i = 0; i < field.dof(); ++i) {
      frame->add_estimates(field.GetP()[i]);
    }    
    return true;
  }

  // These two points are not held by the class.
  RegressorEmsemble<T1, T2>* regressor_emsemble_;
  Image<T2> delta_parameters_;
  ddd::AlgSpec alg_spec_;
};

#endif
