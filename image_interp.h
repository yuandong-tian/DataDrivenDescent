#ifndef _IMAGE_INTERP_H_
#define _IMAGE_INTERP_H_

#include "image.h"

// Image Interface, offer different functionality.
template <class PixelType, class LocationType>
class ImageInterp {
private:
  const ImageView<PixelType> *image_;
public:
  ImageInterp(const ImageView<PixelType> &image) {
      image_ = &image;
  }

  operator const ImageView<PixelType>& () const {
    return *image_;
  }

  // PixelType interpolate_fill_default(LocationType x, LocationType y, PixelType default_value) const {
  //   if (x >= 0 && x < image_->m() - 1 && y >= 0 && y < image_->n() - 1) return interpolate(x, y);
  //   else return default_value;
  // }

  PixelType interpolate_boundary_nearest(LocationType x, LocationType y) const {
    if (x < 0) x = 0;
    else if (x > image_->m() - 1) x = image_->m() - 1;
    if (y < 0) y = 0;
    else if (y > image_->n() - 1) y = image_->n() - 1;

    return interpolate(x, y);
  }

  inline PixelType interpolate(LocationType x, LocationType y) const {
    // Bilinear interpolation. No check.
    const int y_i = floor(y);
    const int x_i = floor(x);
    const LocationType y_r = y - y_i;
    const LocationType x_r = x - x_i;
    
    const PixelType left_top = image_->GetXC(x_i, y_i);
    const PixelType left_bottom = image_->GetXC(x_i, y_i + 1);
    const PixelType right_top = image_->GetXC(x_i + 1, y_i);
    const PixelType right_bottom = image_->GetXC(x_i + 1, y_i + 1);
    
    LocationType c_rb = y_r * x_r;
    LocationType c_rt = x_r - c_rb;  // (1 - y_r) * x_r;
    LocationType c_lb = y_r - c_rb;  // y_r * (1 - x_r);
    LocationType c_lt = 1 - c_rb - c_rt - c_lb;  // (1 - x_r) * (1 - y_r)
    
    return left_top * c_lt + left_bottom * c_lb + right_top * c_rt + right_bottom * c_rb;
  }

  // Call interpolation for each location.
  // void interpolate(const ImageView<PixelType>& img, 
  //                  const ImageView<LocationType>& wx, 
  //                  const ImageView<LocationType>& wy, 
  //                  ImageView<PixelType>* output) {
  //   // Preprocessing:
  //   vector<const PixelType*> scanlines(img.cols());
  //   for (int j = 0; j < img.cols(); ++j) {
  //     scanlines[j] = img.colc(j);
  //   }

  //   // Then interpolate.
  //   for ()
  // }
};

#endif
