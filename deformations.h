#ifndef _DEFORMATIONS_H_
#define _DEFORMATIONS_H_

#include "image_interp.h"
#include "image_filter.h"
#include <glog/logging.h>

// ImageView
struct Point {
  int x, y;
  Point() : x(0), y(0) { }
  Point(int ix, int iy) : x(ix), y(iy) { }
};

template <typename T>
struct PointD {
  T x, y;
  PointD() : x(0), y(0) { }
  PointD(T ix, T iy) : x(ix), y(iy) { } 
};

struct Region {
  int left = 0;
  int top = 0;
  int width = 0;
  int height = 0;

  Region() { }
  Region(int left, int top, int w, int h) : left(left), top(top), width(w), height(h) { } 
};

template <typename T>
class DeformationField {
 public:
  virtual int m() const = 0;
  virtual int n() const = 0;

  // These two functions are supposed to be very slow.
  virtual PointD<T> w(int i, int j) const = 0;
  virtual PointD<T> w_f(T x, T y) const = 0;

  // Reference form. if there is a direct reference, return its pointer, else return nullptr.
  virtual const ImageView<T>* Wx() const = 0;
  virtual const ImageView<T>* Wy() const = 0;

  // Fill-in form. Fillin every entry. If there is reference form, it returns false.
  virtual bool Wx(ImageView<T>& wx) const = 0;
  virtual bool Wy(ImageView<T>& wy) const = 0;

  virtual DeformationField<T>* Inverse() const = 0;
  virtual void Zero() = 0;

  virtual ~DeformationField() { }
};

template <typename T>
class NonparametricField : public DeformationField<T> {
private:
  // We have wx_ and wy_ as dense warping field.
  std::unique_ptr<ImageView<T>> wx_, wy_;
  const T* wx_ptr_;
  const T* wy_ptr_;
  int x_, y_;

public:
  typedef T LocationType;

  // Construction that needs allocation.
  NonparametricField(int m, int n) { 
    wx_.reset(new Image<T>(m, n));
    wy_.reset(new Image<T>(m, n));
  }

  // Construction that does not need allocation.
  NonparametricField(const ImageView<T>& wx, const ImageView<T>& wy) {
    CHECK_EQ(wx.rows(), wy.rows());
    CHECK_EQ(wx.cols(), wy.cols());

    wx_.reset(new ImageView<T>(wx.ShallowCopy()));
    wy_.reset(new ImageView<T>(wy.ShallowCopy()));
  }

  int m() const override { return wx_->m(); }
  int n() const override { return wy_->n(); }

  // Random access.
  PointD<T> w(int i, int j) const override {
    return PointD<T>((*wx_)(i, j), (*wy_)(i, j));
  }
  PointD<T> w_f(T x, T y) const override {
    return PointD<T>((*wx_)(x, y), (*wy_)(x, y));
  }

  // Sequential access.
  // bool reset_iteration() const override {
  //   wx_ptr_ = wx_.ptrc();
  //   wy_ptr_ = wy_.ptrc();
  //   x_ = y_ = 0;
  //   return true;
  // }
  // bool next(PointD<T>* p) const override {
  //   p->x = *wx_ptr_++;
  //   p->y = *wy_ptr_++;
  //   x_ ++;

  //   return true;
  // }

  void Zero() override {
    wx_->Zero();
    wy_->Zero();
  }

  const ImageView<T>* Wx() const override { return wx_.get(); }
  const ImageView<T>* Wy() const override { return wy_.get(); }

  bool Wx(ImageView<T>& wx) const override { return false; }
  bool Wy(ImageView<T>& wy) const override { return false; }

  ImageView<T>& WxMutable() { return *wx_; }
  ImageView<T>& WyMutable() { return *wy_; }

  NonparametricField<T>* Inverse() const override {
    // inverse a nonparametric field..
    const int mm = wx_->m();
    const int nn = wx_->n();

    NonparametricField<T>* res = new NonparametricField<T>(mm, nn);
    if (res == nullptr) return res;

    Image<T> counter(mm, nn);
    Image<T> counter2(mm, nn);
    Image<T> wx(mm, nn);
    Image<T> wy(mm, nn);
 
    res->Zero();
    wx.Zero();
    wy.Zero();
    counter.Zero();

    // for each pixel, find its destination, and blur the image a little bit
    for (int y = 0; y < nn; ++y) {
      const T *wx_col = wx_->colc(y);
      const T *wy_col = wy_->colc(y);

      for (int x = 0; x < mm; ++x) {
        T dx = *wx_col++;
        T dy = *wy_col++;

        int newx = static_cast<int>(x + dx + 0.5);
        int newy = static_cast<int>(y + dy + 0.5);
        if (!wx.indexValid(newx, newy)) continue;

        wx(newx, newy) = -dx;
        wy(newx, newy) = -dy;
        counter(newx, newy) += 1.0;
      }
    }
    // TODO: then we should blur the image and take the local average.
    T sigma = 1.0;
    GaussianSmoothing(wx, sigma, *res->wx_);
    GaussianSmoothing(wy, sigma, *res->wy_);
    GaussianSmoothing(counter, sigma, counter2);

    // Division
    for (int y = 0; y < nn; ++y) {
      T *wx_col = res->wx_->col(y);
      T *wy_col = res->wy_->col(y);
      const T *counter_ptr = counter2.colc(y);

      for (int x = 0; x < mm; ++x) {
        T c = *counter_ptr++ + 1e-4;
        *wx_col++ /= c;
        *wy_col++ /= c;
      }
    }

    return res;
  }
};

// Abstract class: Warping field W(x; p) with parameter p.
template <typename T>
class ParametricField : public DeformationField<T> {
protected:
  int m_, n_;
  vector<T> ps_;

public:
  typedef T LocationType;

  ParametricField(int m, int n, int degree_of_freedom)
   : m_(m), n_(n), ps_(degree_of_freedom, 0) {
  }

  int m() const override { return m_; }
  int n() const override { return n_; }

  // Degrees of freedom. length of params in SetP.
  int dof() const { return (int)ps_.size(); }

  const vector<T> &GetP() const { return ps_; }
  bool SetP(const std::vector<T>& params) {
    if (params.size() != ps_.size()) return false;
    ps_ = params;
    return true;
  }

  bool AddP(const std::vector<T>& params) {
    if (params.size() != ps_.size()) return false;
    for (int i = 0; i < ps_.size(); ++i) {
      ps_[i] += params[i];
    }
    return true;
  }

  bool SetP(const ImageView<T>& params) {
    if (params.rows() != ps_.size() || params.cols() != 1) return false;
    for (int i = 0; i < params.size(); ++i) {
      ps_[i] = params(i, 0);
    }
    return true;
  }

  bool AddP(const ImageView<T>& params) {
    if (params.rows() != ps_.size()) return false;
    for (int i = 0; i < ps_.size(); ++i) {
      ps_[i] += params(i, 0);
    }
    return true;
  }

  // Make all parameters zero.
  void Zero() override { fill(ps_.begin(), ps_.end(), 0.0); }

  // Derivative with respect to the k-th element of p, under current parameter set. 
  // it has two components.
  virtual bool dw(int k, ImageView<T>& dwx, ImageView<T>& dwy) const = 0;

  // Get the parametric inverse of the field, if there is no inverse, return nullptr.
  virtual ParametricField<T> *GroupInverse() const = 0;

  DeformationField<T>* Inverse() const override {
    ParametricField<T>* parametric_inverse = GroupInverse();
    if (parametric_inverse == nullptr) {
      NonparametricField<T> nonparametric(m(), n());
      if (this->Wx() == nullptr) {
        this->Wx(nonparametric.WxMutable());
        this->Wy(nonparametric.WyMutable());
      } else {
        nonparametric.WxMutable().CopyFrom(*this->Wx());
        nonparametric.WyMutable().CopyFrom(*this->Wy());
      }

      return nonparametric.Inverse();
    }
    else return parametric_inverse;
  }
};

// Deformation field.
template <typename T>
class AffineField : public ParametricField<T> {
protected:
  // A = [a_, b_; c_, d_], b = [e_; f_]
  // ps_[0] = a_
  // ps_[1] = b_ 
  // ps_[5] = f_;

  // Deformation: given x, return x' = Ax + b
public:
  typedef ParametricField<T> Base;
  typedef T LocationType;
  using ParametricField<T>::ps_;
  using ParametricField<T>::m_;
  using ParametricField<T>::n_;
  
  AffineField(int m, int n)
    : ParametricField<T>(m, n, 6) {
  }
  AffineField(int m, int n, T a, T b, T c, T d, T e, T f)
    : ParametricField<T>(m, n, 6) {
    ps_[0] = a;
    ps_[1] = b;
    ps_[2] = c;
    ps_[3] = d;
    ps_[4] = e;
    ps_[5] = f;
  }

  // Parametric..
  PointD<T> w(int i, int j) const override {
    return w_f(static_cast<T>(i), static_cast<T>(j));
  }
  PointD<T> w_f(T x, T y) const override {
    return PointD<T>(ps_[0] * x + ps_[1] * y + ps_[4], 
		     ps_[2] * x + ps_[3] * y + ps_[5]);
  }

  const ImageView<T>* Wx() const override { return nullptr; }
  const ImageView<T>* Wy() const override { return nullptr; }

  bool Wx(ImageView<T>& wx) const override {
    if (wx.m() != m_ || wx.n() != n_)
      return false;

    for (int x = 0; x < m_; x++) {
      for (int y = 0; y < n_; y++) {
        wx(x, y) = ps_[0] * x + ps_[1] * y + ps_[4]; 
      }
    }
    return true;
  }

  bool Wy(ImageView<T>& wy) const override { 
    if (wy.m() != m_ || wy.n() != n_)
      return false;

    for (int x = 0; x < m_; x++) {
      for (int y = 0; y < n_; y++) {
        wy(x, y) = ps_[2] * x + ps_[3] * y + ps_[5];
      }
    }
    return true;
  }
  
  // Derivative with respect to the k-th element of p, it has two components.
  bool dw(int k, ImageView<T>& dwx, ImageView<T>& dwy) const override {
    if (dwx.m() != m_ || dwx.n() != n_ || dwy.m() != m_ || dwy.n() != n_ || k < 0 || k >= Base::dof())
      return false;

    dwx.Zero();
    dwy.Zero();
    
    switch (k) {
    case 0:
      // x 
      dwx.FillUsingM();
      break;
    case 1:
      dwx.FillUsingN();
      break;
    case 2:
      dwy.FillUsingM();
      break;
    case 3:
      dwy.FillUsingN();
      break;
    case 4:
      dwx.Fill(1.0);
      break;
    case 5:
      dwy.Fill(1.0);
      break;
    }
    return true;
  }

  AffineField<T> *GroupInverse() const override {
    T a = ps_[0]; 
    T b = ps_[1]; 
    T c = ps_[2]; 
    T d = ps_[3]; 
    T e = ps_[4]; 
    T f = ps_[5]; 

    T delta = (1 + a) * (1 + d) - b * c;
    T inv_a = (1 + d) / delta - 1;
    T inv_d = (1 + a) / delta - 1;
    T inv_b = -b / delta;
    T inv_c = -c / delta;
    
    return new AffineField(m_, n_, inv_a, inv_b, inv_c, inv_d, -e, -f);
  }
};

template <typename T>
class TranslationalField : public ParametricField<T> {
private:
  using ParametricField<T>::ps_;
  using ParametricField<T>::m_;
  using ParametricField<T>::n_;

public:
  TranslationalField(int m, int n) 
      : ParametricField<T>(m, n, 2) {
  }

  PointD<T> w(int i, int j) const override { return PointD<T>(ps_[0], ps_[1]); }
  PointD<T> w_f(T x, T y) const override { return PointD<T>(ps_[0], ps_[1]); } 

  const ImageView<T>* Wx() const override { return nullptr; }
  const ImageView<T>* Wy() const override { return nullptr; }

  bool Wx(ImageView<T>& wx) const override {
    if (wx.m() != m_ || wx.n() != n_)
      return false;

    wx.Fill(ps_[0]);
    return true;
  }

  bool Wy(ImageView<T>& wy) const override { 
    if (wy.m() != m_ || wy.n() != n_)
      return false;

    wy.Fill(ps_[1]);
    return true;
  }
  
  // Derivative with respect to the k-th element of p, it has two components.
  bool dw(int k, ImageView<T>& dwx, ImageView<T>& dwy) const override {
    if (k == 0) {
      dwx.Fill(1.0);
      dwy.Zero();
      return true;
    } 

    if (k == 1) {
      dwx.Zero();      
      dwy.Fill(1.0);
      return true;
    }
    return false;
  }

  TranslationalField<T> *GroupInverse() const override {
    TranslationalField<T>* field = new TranslationalField<T>(m_, n_);
    if (field == nullptr) return nullptr;
    field->ps_[0] = -ps_[0];
    field->ps_[1] = -ps_[1];
    return field;
  }  
};

template <typename T>
class BasesField : public ParametricField<T> {
protected:
  // 
  vector<std::unique_ptr<Image<T>>> bx_;
  vector<std::unique_ptr<Image<T>>> by_;

public:
  typedef ParametricField<T> Base;
  typedef T LocationType;
  using ParametricField<T>::ps_;
  using ParametricField<T>::m_;
  using ParametricField<T>::n_;

  BasesField(int m, int n, int nC) : ParametricField<T>(m, n, nC) {
    bx_.resize(nC);
    by_.resize(nC);

    for (int i = 0; i < nC; ++i) { 
      bx_[i].reset(new Image<T>(m_, n_));
      by_[i].reset(new Image<T>(m_, n_));
    }
  }

  Image<T>& Bx(int k) { return *bx_[k]; }
  Image<T>& By(int k) { return *by_[k]; }
  const Image<T>& Bx(int k) const { return *bx_[k]; }
  const Image<T>& By(int k) const { return *by_[k]; }

  // These two are not efficient.
  PointD<T> w(int i, int j) const override {
    PointD<T> res(0, 0);
    for (int k = 0; k < ps_.size(); k++) {
      res.x += (*bx_[k])(i, j) * ps_[k];
      res.y += (*by_[k])(i, j) * ps_[k];
    }
    return res;
  }

  PointD<T> w_f(T x, T y) const override {
    return w(x, y);
  }

  const ImageView<T>* Wx() const override { return nullptr; }
  const ImageView<T>* Wy() const override { return nullptr; }

  bool Wx(ImageView<T>& wx) const override {
    if (wx.m() != m_ || wx.n() != n_)
      return false;

    wx.Zero();
    // Slow!
    for (int x = 0; x < m_; ++x) {
      for (int y = 0; y < n_; ++y) {
        T& val = wx(x, y);
        for (int k = 0; k < ps_.size(); k++) {
          val += (*bx_[k])(x, y) * ps_[k];
        }
      }
    }
    return true;
  }

  bool Wy(ImageView<T>& wy) const override { 
    if (wy.m() != m_ || wy.n() != n_)
      return false;

    wy.Zero();
    for (int x = 0; x < m_; ++x) {
      for (int y = 0; y < n_; ++y) {
        T& val = wy(x, y);
        for (int k = 0; k < ps_.size(); k++) {
          val += (*by_[k])(x, y) * ps_[k];
        }
      }
    }
    return true;
  }

  bool dw(int k, ImageView<T>& dwx, ImageView<T>& dwy) const override {
    if (k < 0 || k >= Base::dof())
      return false;

    dwx.CopyFrom(*bx_[k]);
    dwy.CopyFrom(*by_[k]);

    return true;
  }

  ParametricField<T> *GroupInverse() const override { return nullptr; }
};

template <typename T>
class LandmarkField : public BasesField<T> {
  // deformation field controlled by landmarks.
protected:
  vector<Point> landmarks_;

  using BasesField<T>::bx_;
  using BasesField<T>::by_;
  using BasesField<T>::m_;
  using BasesField<T>::n_;

  static constexpr T eps = 1e-4;

  static T thinplate_kernel(T x1, T y1, T x, T y) {
    T dx = x1 - x;
    T dy = y1 - y;
    T r2 = dx * dx + dy * dy + eps;

    return r2 * log(r2 + eps);
  }

  void compute_thinplate_matrix() {
    // compute intermediate results for faster computation.
    // Format: ps_ = [delta x1, delta y1, delta x2, delta y2, ..., delta xn, delta yn];
    //
    // Let W(x) = \sum_i c_i (-r_i^2 log r_i^2) = B(x, l)c, where
    // r_i^2 = (x - x_i)^2 + (y - y_i)^2
    // Setting W(landmark) = B(landmark)c = ps_, we can get c = B_L^-1 * ps_.
    // Therefore, given ps_, 
    // we obtain c and thus obtain deformation W(x) = B(x, l)B_L^-1 ps_.
    // So the task here is to compute B(x)B_L^-1:
    const int K = landmarks_.size();

    for (int k = 0; k < 2 * K; ++k) {
      if (bx_[k] == nullptr || by_[k] == nullptr) {
        throw std::runtime_error("Bases are not properly allocated.");
      }
      bx_[k]->Zero();
      by_[k]->Zero();
    }

    Image<T> B(m_*n_, K);
    Image<T> BlBlinv(m_*n_, K);
    Image<T> Bl(K, K);

    for (int k = 0; k < K; ++k) {
      const Point& l = landmarks_[k];
      T* b_column = B.col(k);
      for (int y = 0; y < n_; y++) {
        for (int x = 0; x < m_; x++) {
          *b_column++ = thinplate_kernel(l.x, l.y, x, y);
        }
      }

      // Find the matrix to be inverted.
      for (int k2 = 0; k2 <= k; k2++) {
        const int k2_index = landmarks_[k2].x + m_ * landmarks_[k2].y;

        if (k2_index < 0 || k2_index >= m_*n_) {
          throw std::runtime_error(
            StringPrintf("Landmark (%d, %d) are outside the image (%d, %d)", 
              landmarks_[k2].x, landmarks_[k2].y, m_, n_));
        }

        T v = B(k2_index, k);
        Bl(k2, k) = v;
        Bl(k, k2) = v;
      }
    }

    // Inverse the matrix and matrix Multiplication.
    Multiply(B, Inverse(Bl), BlBlinv);

    // Save it back to the bases.
    for (int k = 0; k < K; ++k) {
      bx_[k]->CopyFrom(ImageView<T>(BlBlinv.col(k), m_, n_));
      by_[k + K]->CopyFrom(ImageView<T>(BlBlinv.col(k), m_, n_));
    }
  }

public:
  typedef T LocationType;

  LandmarkField(int m, int n, const vector<Point>& landmarks)
    : BasesField<T>(m, n, landmarks.size() * 2), landmarks_(landmarks) {
      // cout << "Before thinplate matrix" << endl;
      compute_thinplate_matrix();
  }

  // Set the parameters in the following order:
  // [dx1, dx1, ..., dy1, dy2, ...]
  // bool SetP(const std::vector<T>& params);
};

// Deformation
template <typename T, typename LocationType, template <class U> class DeformationType>
bool WarpBackward(const ImageView<T>& image, const DeformationType<LocationType>& field, ImageView<T>* result) {
  assert(image.rows() == result->rows());
  assert(image.cols() == result->cols());
  assert(image.rows() == field.m());
  assert(image.cols() == field.n());
  
  /*vector<double> params(field.dof());
  field.GetP(params);
  for (int i = 0; i < field.dof(); ++i) {
    if (isnan(params[i])) {
      g_log.PrintInfo("WarpBackward: params[%d] = nan!", i);
      return false;
    }
    }*/

  result->Zero();

  ImageInterp<T, LocationType> image_interp(image);
  // const T default_value = 0.0;

  const ImageView<T>* wx;
  const ImageView<T>* wy;
  bool owned = false;

  /*
  if (field.Wx() != nullptr) {
    wx = field.Wx();
    wy = field.Wy();
  } else {
    Image<PixelType>* wx_init = new Image<PixelType>(image.m(), image.n());
    Image<PixelType>* wy_init = new Image<PixelType>(image.m(), image.n());
    field.Wx(*wx_init);
    field.Wy(*wy_init);

    wx = wx_init;
    wy = wy_init;
    owned = true;
  }
  */

  for (int x = 0; x < image.rows(); ++x) {
    for (int y = 0; y < image.cols(); ++y) {
      //LocationType xc = x + wx->GetXC(x, y);
      //LocationType yc = y + wy->GetXC(x, y);
      const PointD<LocationType> p = field.w(x, y);
      LocationType xc = x + p.x;
      LocationType yc = y + p.y;

      (*result)(x, y) = image_interp.interpolate_boundary_nearest(xc, yc);
      
      //D(if (isnan(xc)) g_log.PrintInfo("xc is nan at (%d, %d)", x, y););
      //D(if (isnan(yc)) g_log.PrintInfo("yc is nan at (%d, %d)", x, y););
      
      //D(if (isnan(result(x, y)))
      //  g_log.PrintInfo("WarpBackward: nan in (%d, %d), size = (%d, %d)", x, y, image.m(), image.n()));
    }
  }
  
  //D(if (result.HasNan())
  //    g_log.PrintInfo("WarpBackward: result has nan entries!"));

  if (owned) {
    delete wx;
    delete wy;
  }

  return true;
}

// Deformation
template <typename T, class LocationType, template <class U> class DeformationType>
bool WarpBackward(const ImageView<T>& image, const DeformationType<LocationType>& field, 
                  const vector<Point>& sample_locs, ImageView<T>* values) {
  assert(image.rows() == field.m());
  assert(image.cols() == field.n());

  CHECK_NOTNULL(values);  
  assert(values->rows() == sample_locs.size());

  ImageInterp<T, LocationType> image_interp(image);
  // const T default_value = 0.0;

  T* values_ptr = values->ptr();
  for (const Point& s : sample_locs) {
      const PointD<LocationType> p = field.w(s.x, s.y);
      LocationType xc = s.x + p.x;
      LocationType yc = s.y + p.y;
      *values_ptr++ = image_interp.interpolate_boundary_nearest(xc, yc);
  }

  return true;
}

template <typename PixelType, class LocationType, template <class U> class DeformationType>
bool WarpForward(const ImageView<PixelType>& image, const DeformationType<LocationType>& field, ImageView<PixelType>* result) {
  CHECK_NOTNULL(result);

  CHECK_EQ(image.m(), result->m());
  CHECK_EQ(image.n(), result->n());
  CHECK_EQ(image.m(), field.m());
  CHECK_EQ(image.n(), field.n());
  
  auto *inv_field = field.Inverse();
  if (inv_field != nullptr) {
    WarpBackward(image, *inv_field, result);
    delete inv_field;
    return true;
  }
  else return false;
}

#endif
