import numpy;
import itertools
import deformation_pylib
import data_driven_descent_pb2;
import glob
import cv2

import sys
import os;

def pairwise_l2_dist(m):
    N, dim = m.shape
    norm_sqr = (m**2).sum(axis=1);
    inner_prod = numpy.dot(m, m.T);
    # ||x_i - x_j||^2 = x_i^2 + x_j^2 - 2*x_i*x_j
    l2_dist = norm_sqr[:,None] + norm_sqr[None,:] - 2*inner_prod;
    return l2_dist;

def compute_lipschitz(delta_i, delta_p):
    # Sort delta_i and delta_p 
    sorted_pair = sorted(zip(delta_i.flat, delta_p.flat), key=lambda x: x[0]);

    # Read from the list
    Gs = [];
    gs = [];
    max_g = 0;
    max_G = sorted_pair[-1][0];

    for G, g in sorted_pair:
        Gs.append(G / max_G);
        gs.append(max(max_g, g));
        max_g = gs[-1];

    return Gs, gs, max_G

def create_smooth_deformation(m, n):
    wx = numpy.random.normal(0, 100.0, (m, n)).astype('f4');
    wy = numpy.random.normal(0, 100.0, (m, n)).astype('f4');
    wx = filter.gaussian_filter(wx, 10)
    wy = filter.gaussian_filter(wy, 10)
    return (wx, wy)

def create_landmarks(m, n, side):
    landmarks = numpy.zeros((2, side, side), dtype='f4')
    for i in range(side):
        ri = (float(i) + 0.5) / side;
        for j in range(side):
            rj = (float(j) + 0.5) / side;
            landmarks[:,i,j] = [ri * m, rj * n];
    return landmarks

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

# Parameter setting
def get_def_spec(img, landmarks, warp_type):
    def_spec = data_driven_descent_pb2.DeformationSpec();
    def_spec.warp_type = data_driven_descent_pb2.FORWARD;
    def_spec.image_width = img.shape[1]
    def_spec.image_height = img.shape[0]

    if warp_type == "affine":
        def_spec.deformation_type = data_driven_descent_pb2.AFFINE;
        def_spec.dof = 6;
    elif warp_type == "landmark":
        def_spec.deformation_type = data_driven_descent_pb2.LANDMARK;
        # landmarks.
        for val in landmarks.flat:
            def_spec.landmarks.append(float(val));
        def_spec.num_landmarks = landmarks.size / landmarks.shape[0];
        def_spec.dof = landmarks.size;

    print img.shape

    return def_spec;

def get_ddd_parameters(dof, distortion_sigma, landmarks):
    alg_spec = data_driven_descent_pb2.AlgSpec();
    alg_spec.num_samples = 3000;
    alg_spec.sigma = distortion_sigma;
    alg_spec.power = 1;
    alg_spec.nearest_neighbor = 10;
    alg_spec.num_iterations = 50;

    # Specify the region.
    region = alg_spec.regions.add();
    region.layer = 0;
    region.left = 0;
    region.top = 0;
    region.width = img.shape[0];
    region.height = img.shape[1];
    #region.subsets.extend(range(landmarks.size));
    region.subsets.extend(range(dof));

    region.universal = True;

    return alg_spec;

def get_hddd_parameters(img, distortion_sigma, landmarks):
    alg_spec = data_driven_descent_pb2.AlgSpec();
    alg_spec.nearest_neighbor = 10;

    # Specify the region.
    num_layer = landmarks.shape[1];
    num_landmark = landmarks.size / landmarks.shape[0];

    margin = 20;
    landmarks_2 = landmarks.reshape((2, landmarks.shape[1] * landmarks.shape[2]));

    for layer in range(num_layer):
        layer_spec = alg_spec.layers.add();

        layer_spec.layer = layer;
        layer_spec.num_iterations = 1;

        magnitude = 0;

        side = num_layer - layer;
        for i in range(layer + 1):
            for j in range(layer + 1):
                region = layer_spec.regions.add();

                region.left = int(landmarks[0,i:i+side,j:j+side].min()) - margin
                region.top = int(landmarks[1,i:i+side,j:j+side].min()) - margin
                right = int(landmarks[0,i:i+side,j:j+side].max()) + margin
                bottom = int(landmarks[1,i:i+side,j:j+side].max()) + margin

                region.left = max(region.left, 0)
                region.top = max(region.top, 0)
                right = min(right, img.shape[0])
                bottom = min(bottom, img.shape[1])

                region.width = right - region.left;
                region.height = bottom - region.top;

                region.num_samples_per_dim = 10;

                region.max_magnitude = min(region.width, region.height) / 2;
                magnitude = max(magnitude, region.max_magnitude);

                for ii in range(i, i+side):
                    for jj in range(j, j+side):
                        index = jj + ii * landmarks.shape[2];
                        region.subsets.extend([index, index + num_landmark]);

                # print "Box (layer %d): [%d %d %d %d]" % (region.layer, region.left, region.top, region.width, region.height);
                # print "Subsets: ", region.subsets
                # print "Sample landmark: landmark[%d] = %s" % (region.subsets[0], str(landmarks_2[:,region.subsets[0]]));
                # print "max_magnitude: ", region.max_magnitude;
        layer_spec.sample_spec.num_samples = 50;
        layer_spec.sample_spec.sigma = magnitude;
        layer_spec.sample_spec.power = 1;
        layer_spec.sample_spec.sample_type = data_driven_descent_pb2.ONLY_TRANSLATION;

    return alg_spec;    

def landmark_pick_subset(landmarks, rect):
    def filter_landmark(landmark, rect):
        return rect[0] <= landmark[0] < rect[2] + rect[0] and rect[1] <= landmark[1] < rect[3] + rect[1];
    return [idx for idx, column in enumerate(landmarks.T) if filter_landmark(column, rect)];

def subset_to_key(subset, max_n):
    s = ["0"] * max_n;
    for i in subset: s[i] = "1";
    return "".join(s);

def subset_subsume(subsets, subset):
    return any([subset.issubset(s) for s in subsets]);

def landmark_get_rect(landmarks, subset, minsize=0):
    landmark_subset = landmarks[:,list(subset)];
    mins = landmark_subset.min(axis=1);
    maxs = landmark_subset.max(axis=1);

    rect = [mins[0], mins[1], maxs[0] - mins[0], maxs[1] - mins[1]];
    if rect[2] < minsize[0]: 
        margin = minsize[0] - rect[2];
        rect[0] -= margin / 2;
        rect[2] = minsize[0];

    if rect[3] < minsize[1]: 
        margin = minsize[1] - rect[3];
        rect[1] -= margin / 2;
        rect[3] = minsize[1];

    return map(int, rect);

def get_hddd_parameters_fixed_layer(img, distortion_sigma, landmarks, num_layer, ratio):
    alg_spec = data_driven_descent_pb2.AlgSpec();
    alg_spec.nearest_neighbor = 10;

    # Specify the region.
    num_landmark = landmarks.size / landmarks.shape[0];
    landmarks_flattern = landmarks.reshape((2, num_landmark));

    # 
    h, w, nchannel = img.shape;
    curr_h = float(h);
    curr_w = float(w);

    for layer in range(num_layer - 1):
        layer_spec = alg_spec.layers.add();

        layer_spec.layer = layer;
        layer_spec.num_iterations = 1;
        layer_spec.num_samples_per_dim = 50;

        subsets = [];

        max_w = max_h = None;

        # Then for each landmark, generate a rectangle.
        for i in range(num_landmark):
            x = landmarks_flattern[0,i]
            y = landmarks_flattern[1,i]

            rect = (x - 1, y - 1, int(curr_w), int(curr_h));
            subset = set(landmark_pick_subset(landmarks_flattern, rect));
            if any([len(subset.intersection(s)) > min(len(s), len(subset)) * 0.9 for s in subsets]): continue;

            subsets.append(subset);
            rect = landmark_get_rect(landmarks_flattern, subset, minsize=(20, 20));
            # print subset;

            # Rectangle is too small compared to the largest size so far, skip.
            if max_w is not None and max_h is not None and rect[2] < max_w * ratio and rect[3] < max_h * ratio: continue;
            max_w = max(max_w, rect[2]);
            max_h = max(max_h, rect[3]);

            region = layer_spec.regions.add();
            region.left = rect[0];
            region.top = rect[1];
            region.width = rect[2];
            region.height = rect[3];

            region.max_magnitude = min(region.width, region.height) / 2;
            region.subsets.extend(subset);
            region.subsets.extend([n + num_landmark for n in subset]);

        # import pdb;
        # pdb.set_trace()
        # Get associated size.
        print "Layer %d: #region = %d" % (layer, len(layer_spec.regions));

        layer_spec.sample_spec.num_samples = 50;
        layer_spec.sample_spec.sigma = (curr_h + curr_w) / 40;
        layer_spec.sample_spec.power = 1;
        layer_spec.sample_spec.sample_type = data_driven_descent_pb2.ONLY_TRANSLATION;        

        curr_h = curr_h * ratio;
        curr_w = curr_w * ratio;

                # print "Box (layer %d): [%d %d %d %d]" % (region.layer, region.left, region.top, region.width, region.height);
                # print "Subsets: ", region.subsets
                # print "Sample landmark: landmark[%d] = %s" % (region.subsets[0], str(landmarks_2[:,region.subsets[0]]));
                # print "max_magnitude: ", region.max_magnitude;

    # Last layer, 
    margin = 10;
    layer_spec = alg_spec.layers.add();
    layer_spec.layer = num_layer - 1;
    layer_spec.num_samples_per_dim = 50;

    for i in range(num_landmark):
        x = landmarks_flattern[0,i]
        y = landmarks_flattern[1,i]

        region = layer_spec.regions.add();
        region.left = int(x - margin);
        region.top = int(y - margin);
        region.width = 2 * margin + 1;
        region.height = 2 * margin + 1;

        region.max_magnitude = min(region.width, region.height) / 2;
        region.subsets.extend([i, i + num_landmark]);

    layer_spec.sample_spec.num_samples = 50;
    layer_spec.sample_spec.sigma = margin / 2;
    layer_spec.sample_spec.power = 1;
    layer_spec.sample_spec.sample_type = data_driven_descent_pb2.ONLY_TRANSLATION;        

    # print "Total #region = %d" % len(alg_spec.regions);

    return alg_spec;

def load_dataset(directory):
    # Load a dataset
    template = cv2.imread(os.path.join(directory, 'template.png')).astype('f4');

    # Load deformations
    f = open(os.path.join(directory, "landmark.txt"));
    nx, ny = f.readline().split();
    nlandmark, = f.readline().split();

    nx = int(nx);
    ny = int(ny);
    nlandmark = int(nlandmark);

    landmarks = numpy.zeros((2, nlandmark), dtype='f4');
    counter = 0;
    for line in f:
        x, y = line.split();
        landmarks[:,counter] = [x, y];
        counter += 1;
    landmarks = landmarks.reshape((2, nx, ny));

    # Load images.
    m, n, nchannel = template.shape
    filenames = glob.glob(os.path.join(directory, "img*.png"));
    imgs = numpy.zeros((m, n, nchannel, len(filenames)), dtype='f4')

    for idx, f in enumerate(filenames):
        imgs[:,:,:,idx] = cv2.imread(f);

    print "Template size: ", template.shape;
    print "Image size: ", imgs.shape

    return template, landmarks, imgs