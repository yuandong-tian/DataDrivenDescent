#!/usr/bin/python
from matplotlib import pyplot as plt
# from skimage import data, io, filter, transform
import cv2
import cProfile;
import cPickle

import numpy;
import cProfile;
import cPickle
import itertools
import deformation_pylib
import data_driven_descent_pb2;
import distortion_util

import sys

def plot_landmarks(ax, landmarks0, delta=None):
    if delta is not None:
        landmarks = landmarks0 + delta;
    else:
        landmarks = landmarks0;

    landmarks = landmarks.reshape((2, landmarks.shape[1] * landmarks.shape[2]))

    # From landmark size to obtain the radius.
    max_range = numpy.max(landmarks)
    radius = max_range / 100;
    for k, landmark in enumerate(landmarks.T):
        #x = tuple(landmark);
        x = (landmark[0], landmark[1])
        ax.add_patch(plt.Circle(x, radius, color='red', linewidth=2, fill=False));
        ax.text(x[0], x[1], str(k))

def dump_deformed_images(img, landmarks, deform_arg, N):
    d, nlandmark = landmarks.shape;
    m, n, channel = img.shape;

    ps = numpy.zeros((N, d*nlandmark), dtype='f4')
    deforms = numpy.zeros((N, m*n*channel), dtype='f4')

    for i in range(N):
        if i % 500 == 0: print i
        deform = numpy.zeros(img.shape, dtype='f4');    
        p = numpy.random.normal(0, 3.0, landmarks.shape).astype('f4');
        # p[i] = numpy.zeros((2, 25), dtype='f4');
        # x_offset = (2 * (i % 2) - 1) * 15
        # y_offset = (i >= 2 and 1 or -1) * 15
        # p[i][0,:] = x_offset
        # p[i][1,:] = y_offset
        deformation_pylib.SetLandmarkShifts(p)
        deformation_pylib.DeformImageWithLandmarks(img, deform, deform_arg)
        # texts[i] = "x: %d, y: %d" % (x_offset, y_offset)
        deforms[i, :] = deform.flat;
        ps[i, :] = p.flat;

    return ps, deforms

def selector(ps, sel_label, sel_values, sel_thres):
    diff = ps[:,sel_label] - sel_values;
    # filter the samples with small difference.
    sel = numpy.mean(diff**2, axis=1) < sel_thres * sel_thres

    print "Selected sample = %d/%d" % (sel.sum(), ps.shape[0]);
    return sel;

def verify_variance(sel, reps):
    # Check variance within the set.
    mean_label = numpy.mean(reps[sel,:], axis=0);
    diff_label = reps[sel,:] - mean_label;
    return numpy.mean(numpy.mean(diff_label**2, axis=1))

def variance_verify_deformation(img, landmarks, filename):
    sys.path.append('../nn')
    import nn_lib

    net = nn_lib.Net();
    # net.Load("../nn/deformation_model_10-12-2014_12-54-16.bin")
    net.Load(filename);

    N = 50000;
    sel_label = range(0, 25, 4) + range(25, 50, 4);
    sel_label_other = range(2, 25, 4) + range(27, 50, 4);

    all_network_dims = eval(net.GetAllLayerOutputDims());
    print all_network_dims

    layer_outputs = [];
    for dims in all_network_dims:
        mat_size = dims[0] * dims[1] * dims[2];
        layer_outputs.append(numpy.zeros((N, mat_size), dtype='f4'))

    # Dump all responses.
    ps, deforms = dump_deformed_images(img, landmarks, 'backward', N)
    net.DumpAllResponses(deforms, layer_outputs);
    # net.EvaluateRegression(deforms, labels_est);

    threshold = 2.0;
    sel_data_label = selector(ps, sel_label, 0, threshold);
    sel_data_other = selector(ps, sel_label_other, 0, threshold);

    for i in range(len(all_network_dims)):
        var_label = verify_variance(sel_data_label, layer_outputs[i]);
        var_other = verify_variance(sel_data_other, layer_outputs[i]);

        print "Variance_label in layer %d = %f" % (i, var_label)
        print "Variance_other in layer %d = %f" % (i, var_other)

def lipschitz_verify_deformation(img, landmarks, filename):
    sys.path.append('../nn')
    import nn_lib

    net = nn_lib.Net();
    # net.Load("../nn/deformation_model_10-12-2014_12-54-16.bin")
    net.Load(filename);

    sel_label = range(0, 25, 4) + range(25, 50, 4);

    # Dump all responses.
    all_network_dims = eval(net.GetAllLayerOutputDims());
    print all_network_dims

    # build a list of numpy arrays.
    N = 2000;
    layer_outputs = [];
    for dims in all_network_dims:
        mat_size = dims[0] * dims[1] * dims[2];
        layer_outputs.append(numpy.zeros((N, mat_size), dtype='f4'))

    ps, deforms = dump_deformed_images(img, landmarks, 'backward', N)
    net.DumpAllResponses(deforms, layer_outputs);

    sel_label = range(0, 25, 4) + range(25, 50, 4);
    #labels = (ps[:,sel_label] > 0).astype('f4');
    labels = 1.0 / (1.0 + numpy.exp(-ps[:,sel_label]));    

    # For each layer, find pairwise distances.
    pw_images = pairwise_l2_dist(deforms);
    nlayer = len(all_network_dims);
    pw_labels = pairwise_l2_dist(labels);

    colors = ['r', 'g', 'b', 'c'];
    styles = ['-', '--'];
    cs = list(itertools.product(styles, colors))

    for i in range(len(all_network_dims)):
        # print "Size at layer " + str(i) + ":" + str(layer_outputs[i].shape);
        pw_layer_i = pairwise_l2_dist(layer_outputs[i]);
        Gs, gs, max_G = compute_lipschitz(pw_layer_i, pw_labels)
        print "layer %d: max_G = %f" % (i, max_G);
        plt.plot(Gs, gs, cs[i][0] + cs[i][1], label="Layer %d" % i);

    Gs, gs, max_G = compute_lipschitz(pw_labels, pw_labels)
    print "GroundTruth layer: max_G = %f" % max_G;
    plt.plot(Gs, gs, 'b-', label="GroundTruth")

    Gs, gs, max_G = compute_lipschitz(pw_images, pw_labels)
    print "Input image: max_G = %f" % max_G;
    plt.plot(Gs, gs, 'k-', label="InputImage")

    plt.legend();
    plt.show();

    import pdb
    pdb.set_trace()

    # Layers
    # print "--------------------"
    # print -numpy.log(1.0 / label_est - 1.0)
    # print p.flat[sel_label]
    # print "--------------------"

    # print (label_est > 0.5).astype('i4')
    # print (p.flat[sel_label] > 0).astype('i4')

def visualize_deformation_forward(img, landmarks, distortion_sigma):
    print "deforming"
    N = 4;
    ps = numpy.zeros((N, 2*nSide**2), dtype='f4')
    deforms = numpy.zeros((N, m*n*channel), dtype='f4')
    for i in range(N):
        deform = numpy.zeros(img.shape, dtype='f4');    
        p = numpy.random.normal(0, distortion_sigma, landmarks.shape).astype('f4');
        deformation_pylib.SetLandmarkShifts(p);
        deformation_pylib.DeformImageWithLandmarks(img, deform, "forward");
        deforms[i, :] = deform.flat;
        ps[i, :] = p.flat;

    print "showimage"
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')

    count = 0;
    for i in range(2):
        for j in range(2):
            deform = deforms[count, :]
            deform.shape = (m, n, channel)
            p = ps[count, :].reshape((2, nSide, nSide))
            axes[i][j].imshow(deform, interpolation='nearest')
            plot_landmarks(axes[i][j], landmarks, delta=p)
            count += 1;
    plt.show();

def visualize_deformation_backward(img, landmarks, distortion_sigma):
    print "deforming"
    N = 2;
    ps = numpy.zeros((N, 2*nSide**2), dtype='f4')
    deforms = numpy.zeros((N, m*n*channel), dtype='f4')
    for i in range(N):
        deform = numpy.zeros(img.shape, dtype='f4');    
        p = numpy.random.normal(0, distortion_sigma, landmarks.shape).astype('f4');
        deformation_pylib.SetLandmarkShifts(p);
        deformation_pylib.DeformImageWithLandmarks(img, deform, "backward");
        deforms[i, :] = deform.flat;
        ps[i, :] = p.flat;

    print "showimage"
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')

    count = 0;
    for i in range(2):
        deform = deforms[i, :]
        deform.shape = (m, n, channel)
        p = ps[i, :].reshape((2, nSide, nSide))

        axes[i][0].imshow(deform, interpolation='nearest')
        plot_landmarks(axes[i][0], landmarks)
        axes[i][1].imshow(img, interpolation='nearest')
        plot_landmarks(axes[i][1], landmarks, delta=p)
    plt.show();

def run_ddd(deform):
    return data_driven_descent_pb2.Result.FromString(deformation_pylib.EstimationWithDDD(deform));

def test_data_driven_descent(img, landmarks, distortion_sigma):
    # Landmark is set
    def_spec = distortion_util.get_def_spec(img, landmarks, "landmark")
    #alg_spec = get_ddd_parameters(def_spec.dof, distortion_sigma, landmarks)
    alg_spec = distortion_util.get_hddd_parameters_fixed_layer(img, distortion_sigma, landmarks, 7, 0.7);
    alg_spec.dump_intermediate = True;

    deformation_pylib.InitializeDeformation(def_spec.SerializeToString());

    print "GenerateDDDSamples"
    deformation_pylib.GenerateDDDSamples(img, alg_spec.SerializeToString());

    while True:
        deform = numpy.zeros(img.shape, dtype='f4');    
        p = numpy.random.normal(0, distortion_sigma, (1, def_spec.dof)).astype('f4');        
        #p = numpy.array([5.0] * def_spec.dof).astype('f4');                
        deformation_pylib.SetParameters(p);
        deformation_pylib.DeformImage(img, deform);
        import pdb;
        pdb.set_trace();

        print "Run Data-Driven Descent"
        #cProfile.runctx("result = run_ddd(deform)", {"deform" : deform, "run_ddd" : run_ddd}, {});
        result = run_ddd(deform);

        # print the result
        print "Gt:", p
        print "Estimated:", result.estimates

        # show the debug images
        rows = 4;
        cols = 4;

        fig, axes = plt.subplots(rows, cols, sharex='col', sharey='row')

        for frame in result.frames:
            count = frame.t;
            # if frame.t % 2 == 1: continue;
            # if c >= 25: break;

            rectified_img = numpy.array(frame.representation);
            rectified_img.shape = (img.shape[0], img.shape[1], 3);

            row = count / cols;
            col = count % cols;

            #axes[row][col].imshow(rectified_img, interpolation='nearest')
            axes[row][col].imshow(deform, interpolation='nearest')
            axes[row][col].set_title("iteration = %d" % frame.t);

            p = numpy.array(frame.estimates).reshape(landmarks.shape);
            plot_landmarks(axes[row][col], landmarks, delta=p)

        axes[-1][-1].imshow(img, interpolation='nearest');
        axes[-1][-1].set_title("Template");
        plot_landmarks(axes[-1][-1], landmarks)

        plt.show();
        key = raw_input("Press Enter to continue...")
        print int(key)
        if key != '\n': break;

#cProfile.run = eval
scale = 4;

img = cv2.imread("test2.png");
img = img[:,:,0:3]
img = cv2.resize(img, (img.shape[0] / scale, img.shape[1] / scale)).astype('f4')
img = numpy.ascontiguousarray(img)
print img.shape
print img.dtype

# cv2.imshow("Test2", img);
# cv2.waitKey()
# sys.exit(0);

m, n, channel = img.shape;

print "Set Landmarks"
nSide = 3;
landmarks = distortion_util.create_landmarks(m, n, nSide)
# deformation_pylib.SetLandmarks(m, n, landmarks.reshape((2, nSide*nSide)))

#visualize_deformation_backward(img, landmarks, 6.0)
#verify_deformation_prediction("../nn/deformation_backward_model_10-14-2014_22-47-04.bin");
#verify_deformation_prediction(img, landmarks, "./deformation_backward_model_deformation_backward-10-15-2014_21-20-24-deformation.bin");
#variance_verify_deformation(img, landmarks, "./deformation_backward_model_deformation_backward-10-15-2014_21-20-24-deformation.bin")

distortion_sigma = 10.0;
#distortion_sigma = 0.5;
test_data_driven_descent(img, landmarks, distortion_sigma)

# ps, deforms = dump_deformed_images(img, landmarks, "forward", 5000)
# Save the training samples.
# numpy.savez(open("deformation_data_" + deform_arg + ".bin", "w"), deforms=deforms, ps=ps)
