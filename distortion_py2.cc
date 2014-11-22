#include "py_common.h"
#include "deformations.h"
#include "data_driven_descent.h"
#include "data_driven_descent.pb.h"

#include <google/protobuf/text_format.h>

using namespace std;

PyMODINIT_FUNC initdeformation_pylib(void); /* Forward */

int main(int argc, char **argv) {
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    google::InitGoogleLogging(argv[0]);

    /* Add a static module */
    initdeformation_pylib();

    /* Exit, cleaning up the interpreter */
    Py_Exit(0);

    /*NOTREACHED*/
    return 0;
}

/* 'self' is not used */
static PyObject* DeformImageWithField(PyObject *self, PyObject* args) {
    PyArrayObject* img;
    PyArrayObject* wx;
    PyArrayObject* wy;
    PyArrayObject* output;

    const char* str;
    if (!PyArg_ParseTuple(args, "O!O!O!O!s", 
        &PyArray_Type, &img, &PyArray_Type, &wx, 
        &PyArray_Type, &wy, &PyArray_Type, &output, &str))
        return NULL;

    ImageView<PixelType> img_view(py_convert<PixelType>(img, 1));
    ImageView<PixelType> output_view(py_convert<PixelType>(output, 1));

    ImageView<float> wx_view(py_convert<float>(wx));
    ImageView<float> wy_view(py_convert<float>(wy));

    // cout << "Start deformation.." << endl;

    NonparametricField<float> field(wx_view, wy_view);

    string arg_string = str;
    if (arg_string == "forward") {
        WarpForward(img_view, field, &output_view);
    } else if (arg_string == "backward") {
        WarpBackward(img_view, field, &output_view);
    }

    // cout << "End deformation.." << endl;
    Py_INCREF(Py_None);
    return Py_None;

    // return Py_BuildValue("s#", schedules_str.c_str(), schedules_str.size());
}

// Samples.
std::unique_ptr<Representer<PixelType>> representer_;
std::unique_ptr<ParametricField<float>> translation_field_;
std::unique_ptr<ParametricField<float>> deformation_field_;
std::unique_ptr<FieldServer<float>> field_server_;
std::unique_ptr<SampleGenerator<PixelType, float>> generator_;
std::unique_ptr<RegressorEmsemble<PixelType, float>> regressor_emsemble_;
std::unique_ptr<DataDrivenDescent<PixelType, float>> ddd_estimator_;

ddd::AlgSpec alg_spec_;
ddd::DeformationSpec deformation_spec_;

// Initialize the deformation, input is a DeformationSpec proto.
static PyObject * InitializeDeformation(PyObject *self, PyObject* args) {
    const char* spec_str;
    int spec_size;

    if (!PyArg_ParseTuple(args, "s#", &spec_str, &spec_size)) {
        return NULL;
    }

    string spec;
    spec.assign(spec_str, spec_size);
    deformation_spec_.ParseFromString(spec);

    cout << "Image size: " << deformation_spec_.image_width() << ", " << deformation_spec_.image_height() << endl;

    if (deformation_spec_.deformation_type() == ddd::AFFINE) {
        deformation_field_.reset(
            new AffineField<float>(
                deformation_spec_.image_width(), 
                deformation_spec_.image_height()));
    } else if (deformation_spec_.deformation_type() == ddd::LANDMARK) {
        const int nlandmark = deformation_spec_.num_landmarks();
        if (nlandmark * 2 != deformation_spec_.landmarks_size()) {
            throw std::runtime_error("Landmark sizes are inconsistent..");
        }

        ImageView<const float> landmarks_view(
            deformation_spec_.landmarks().data(), nlandmark, 2);

        // cout << "Copying landmarks" << endl;
        vector<Point> landmarks(nlandmark);
        for (int i = 0; i < nlandmark; i++) {
            landmarks[i].x = landmarks_view(i, 0);
            landmarks[i].y = landmarks_view(i, 1);
        }
        deformation_field_.reset(
            new LandmarkField<float>(
                deformation_spec_.image_width(), 
                deformation_spec_.image_height(), landmarks));        
    }

    // Also initialize the translational deformation
    translation_field_.reset(
        new TranslationalField<float>(
            deformation_spec_.image_width(), 
            deformation_spec_.image_height()));

    Py_INCREF(Py_None);
    return Py_None;
}

// Set deformation parameters. Input is a vector of parameters.
// The length of the vector must match with degrees of freedom in deformation.
static PyObject * SetParameters(PyObject *self, PyObject* args) {
    if (deformation_field_ == nullptr) {
        throw std::runtime_error("DeformationField is not initialized..");
    }

    PyArrayObject* p;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &p))
        return NULL;

    ImageView<float> p_view(py_convert<float>(p));
    if (p_view.rows() != deformation_field_->dof()) {
        throw std::runtime_error("The number of parameters are different from the degrees of freedom in the deformation.");
    }

    // cout << "Setting parameters.." << endl;
    deformation_field_->SetP(p_view);
    // cout << "End deformation.." << endl;

    Py_INCREF(Py_None);
    return Py_None;

    // return Py_BuildValue("s#", schedules_str.c_str(), schedules_str.size());
}

// Set translational parameters.
static PyObject * SetTranslationalParameters(PyObject *self, PyObject* args) {
    if (translation_field_ == nullptr) {
        throw std::runtime_error("DeformationField is not initialized..");
    }

    float vx, vy;
    if (!PyArg_ParseTuple(args, "ff", &vx, &vy))
        return NULL;

    // cout << "Setting parameters.." << endl;
    translation_field_->SetP({vx, vy});
    // cout << "End deformation.." << endl;

    Py_INCREF(Py_None);
    return Py_None;

    // return Py_BuildValue("s#", schedules_str.c_str(), schedules_str.size());
}

// Deform the image with given parameters.
// Input: image and output 
static PyObject * DeformImage(PyObject *self, PyObject* args) {
    if (deformation_field_ == nullptr) {
        throw std::runtime_error("LandmarkField is not initialized..");
    }

    PyArrayObject* img;
    PyArrayObject* output;
    // Input: image, output
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img, &PyArray_Type, &output))
        return NULL;

    ImageView<PixelType> img_view(py_convert<PixelType>(img, 1));
    ImageView<PixelType> output_view(py_convert<PixelType>(output, 1));

    // cout << "Start deformation.." << endl;
    if (deformation_spec_.warp_type() == ddd::FORWARD) {
        WarpForward(img_view, *deformation_field_, &output_view);
    } else if (deformation_spec_.warp_type() == ddd::BACKWARD) {
        WarpBackward(img_view, *deformation_field_, &output_view);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* GenerateDDDSamples(PyObject* self, PyObject* args) {
    // Generate a few DDD samples.
    PyArrayObject* img;
    const char* spec_str;
    int spec_size;

    // Arguments: template, algorithm specification.
    if (!PyArg_ParseTuple(args, "O!s#", &PyArray_Type, &img, &spec_str, &spec_size))
        return NULL;

    string spec;
    spec.assign(spec_str, spec_size);

    if (!alg_spec_.ParseFromString(spec)) {
        throw std::runtime_error("Sampling Specification cannot be parsed..");   
    }

    ImageView<PixelType> img_view(py_convert<PixelType>(img, 1));

    // ParametricField<float>* pf = 
    //     alg_spec_.sample_type == ddd::ONLY_TRANSLATION ? translation_field_.get() : 

    field_server_.reset(new FieldServer<float>(deformation_field_.get()));
    generator_.reset(new SampleGenerator<PixelType, float>(field_server_.get()));

    representer_.reset(new Representer<PixelType>(
        deformation_spec_.image_width(), deformation_spec_.image_height(), 
        alg_spec_.layers_size(), alg_spec_.blur_sigma()));

    regressor_emsemble_.reset(new RegressorEmsemble<PixelType, float>());
    regressor_emsemble_->GenerateAndTrain(img_view, representer_.get(), generator_.get(), deformation_spec_, alg_spec_);
    ddd_estimator_.reset(new DataDrivenDescent<PixelType, float>(regressor_emsemble_.get(), alg_spec_));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* EstimationWithDDD(PyObject* self, PyObject* args) {
    if (deformation_field_ == nullptr) {
        throw std::runtime_error("LandmarkField is not initialized..");
    }
    if (ddd_estimator_ == nullptr) {
        throw std::runtime_error("DataDrivenDescent estimator is not initialized..");        
    }
    if (regressor_emsemble_ == nullptr) {
        throw std::runtime_error("RegressorEmsemble is not initialized..");        
    }    

    PyArrayObject* deform;
    // Arguments: deformed image.
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &deform))
        return NULL;

    ImageView<PixelType> deform_view(py_convert<PixelType>(deform, 1));

    if (deform_view.rows() != deformation_spec_.image_width() || 
        deform_view.cols() != deformation_spec_.image_height()) {
        throw std::runtime_error("Size of deformation is not the same as template.");
    }

    // cout << "Start deformation.." << endl;
    // Then estimate things with data-driven descent.
    ddd::Result result;
    ddd_estimator_->Estimate(deform_view, representer_.get(), generator_.get(), deformation_spec_.warp_type(), &result);

    LOG(INFO) << "result.SerializeToString" << endl;
    string output_result;
    result.SerializeToString(&output_result);

    return Py_BuildValue("s#", output_result.c_str(), output_result.size());
}

static PyMethodDef deformation_pylib_methods[] = {
    { "DeformImageWithField", DeformImageWithField, METH_VARARGS, "Deform the image" },
    { "InitializeDeformation", InitializeDeformation, METH_VARARGS, "Set the landmark locations of the image" },
    { "SetParameters", SetParameters, METH_VARARGS, "Set the deformation parameters."},
    { "DeformImage", DeformImage, METH_VARARGS, "Set landmark shifts." },
    { "GenerateDDDSamples", GenerateDDDSamples, METH_VARARGS, "Generate Samples for Data-driven Descent."}, 
    { "EstimationWithDDD", EstimationWithDDD, METH_VARARGS, "Estimate the deformation with Data-Driven Descent."}, 
    { NULL, NULL, 0, NULL }           /* sentinel */
};

PyMODINIT_FUNC initdeformation_pylib(void) {
    PyImport_AddModule("deformation_pylib");
    Py_InitModule("deformation_pylib", deformation_pylib_methods);

    import_array();
}