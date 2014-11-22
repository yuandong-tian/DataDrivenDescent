#include "py_common_boost.h"

#include "deformation.h"

using namespace std;

typedef float PixelType[3];

class DistortionWrapper {
public:
    void DeformImageBackword(py::object img, py::object wx, py::object wy, py::object output) {
        ImageView<PixelType> img_view(py_convert(img, 1));
        ImageView<PixelType> output_view(py_convert(output, 1));

        ImageView<float> wx_view(py_convert(wx));
        ImageView<float> wy_view(py_convert(wy));

        NonparametricField field(wx_view, wy_view);
        WarpBackward(img_view, field, output_view);
    }

    void DeformImageForward(py::object img, py::object wx, py::object wy, py::object output) {
        ImageView<PixelType> img_view(py_convert(img, 1));
        ImageView<PixelType> output_view(py_convert(output, 1));

        ImageView<float> wx_view(py_convert(wx));
        ImageView<float> wy_view(py_convert(wy));

        NonparametricField field(wx_view, wy_view);
        WarpForward(img_view, field, output_view);
    }    
};

BOOST_PYTHON_MODULE(distortion_pylib) {
    // below, we prepend an underscore to methods that will be replaced
    // in Python
    py::class_<SchedulerWrapper>("DistortionWrapper")
       .def("DeformImage", &SchedulerWrapper::MakeSchedule)
    ;
}