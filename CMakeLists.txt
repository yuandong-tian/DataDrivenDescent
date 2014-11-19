cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")
include(UtilProtobuf)

find_package(PythonLibs REQUIRED)
find_package(numpy REQUIRED)
find_package(Protobuf REQUIRED)
find_package(opencv REQUIRED)

# MESSAGE(STATUS "OPENCV: " ${OpenCV_LIBS})
MESSAGE(STATUS "OPENCV: " ${OpenCV_INCLUDE_DIRS})

PROTOBUF_GENERATE_CPP_PYTHON(PROTO_SRCS PROTO_HDRS data_driven_descent.proto)

include_directories(${PROTOBUF_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
    "../common" "../image" "../png" "../cblas/include" "../lapack-3.5.0/lapacke/include")

set(the_target "deformation_pylib")                                              
set(srcs distortion_py2.cc ${PROTO_SRCS})
add_executable(${the_target} ${srcs})

# SET(OPENCV_CPP_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect 
# opencv_photo opencv_nonfree opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_stitching opencv_videostab)
# set(opencv_lib )

target_link_libraries(${the_target} glog ${PYTHON_LIBRARIES} ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(${the_target} PROPERTIES 
    OUTPUT_NAME "deformation_pylib.so" 
    LINK_FLAGS "-shared -fPIC" 
    COMPILE_FLAGS "-O3 -msse3 -std=c++0x")
