#!/usr/bin/env python

'''
Multitarget planar tracking
==================

Example of using features2d framework for interactive video homography matching.
ORB features and FLANN matcher are used. This sample provides PlaneTracker class
and an example of its usage.

video: http://www.youtube.com/watch?v=pzVbhxx6aog

Usage
-----
deformation_tracker.py [<video source>]

Keys:
   SPACE  -  pause video
   c      -  clear targets

Select a textured planar object to track by drawing a box with a mouse.
'''

import numpy as np
import cv2

# local modules
import video
import common
import distortion_util

import deformation_pylib
import data_driven_descent_pb2;

class DeformationTracker:
    def __init__(self):
        self.rect = None;
        self.deltas = None;
        self.trackers = None;
        self.nSide = 7;
        self.sigma = 10.0;
        self.records = {};

    def crop(self, img, rect):
        return np.ascontiguousarray(img[rect[1]:rect[3],rect[0]:rect[2],:].astype('f4'));

    def set_target(self, img, rect, landmarks=None):
        # rect: [x0, y0, x1, y1]
        self.template = self.crop(img, rect);
        self.rect = list(rect);

        m, n, nchannel = self.template.shape;
        if landmarks is not None:
            self.landmarks = landmarks.copy();
        else:
            self.landmarks = distortion_util.create_landmarks(m, n, self.nSide)

        self.landmarks_global = self.landmarks.copy();
        self.landmarks_global[0,:,:] += rect[0]
        self.landmarks_global[1,:,:] += rect[1]

        def_spec = distortion_util.get_def_spec(self.template, self.landmarks, "landmark");
        alg_spec = distortion_util.get_hddd_parameters_fixed_layer(self.template, self.sigma, self.landmarks, 7, 0.7)
        # alg_spec = distortion_util.get_hddd_parameters(self.template, self.sigma, self.landmarks);

        deformation_pylib.InitializeDeformation(def_spec.SerializeToString());

        print "GenerateDDDSamples"
        deformation_pylib.GenerateDDDSamples(self.template, alg_spec.SerializeToString());

    def track(self, frame, index=None):
        if not self.rect: return;
        if index in self.records: 
            self.trackers = self.records[index];
            return;
        result = data_driven_descent_pb2.Result.FromString(deformation_pylib.EstimationWithDDD(self.crop(frame, self.rect)));
        # Output the landmarks.
        self.deltas = np.array(result.estimates).reshape(self.landmarks.shape);
        self.trackers = self.landmarks_global + self.deltas;
        if index: self.records.update({index : self.trackers});

    def visualize(self, vis):
        if self.trackers is not None:
            # Draw the circles..
            for i in range(self.trackers.shape[1]):
                for j in range(self.trackers.shape[2]):
                    x = int(self.trackers[0,i,j]);
                    y = int(self.trackers[1,i,j]);
                    cv2.circle(vis, (x, y), 5, (255, 255, 255))

class App:
    def __init__(self, cap, landmarks=None, template=None):
        self.cap = cap
        self.frame = None
        self.paused = False
        self.tracker = DeformationTracker()
        self.landmarks = landmarks;

        cv2.namedWindow('plane')

        self.rect_sel = common.RectSelector('plane', self.on_rect)
        if self.landmarks is not None and template is not None:
            h, w, nchannel = template.shape;
            self.tracker.set_target(template, (0, 0, w, h), landmarks=landmarks)

    def on_rect(self, rect):
        # rect: [x0, y0, x1, y1]
        if rect[2] - rect[0] < 10 or rect[3] - rect[1] < 10: return;

        rect2 = (rect[0], rect[1], rect[2],  rect[2] - rect[0] + rect[1]);
        self.tracker.set_target(self.frame, rect2)

    def run(self, use_camera):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                index = self.cap.index if not use_camera else None;

                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame;

            vis = self.frame.copy()
            if playing:
                self.tracker.track(self.frame, index);
                self.tracker.visualize(vis)

            if self.rect_sel.dragging:
                self.rect_sel.draw(vis)
            cv2.imshow('plane', vis)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break

    def play(self):
        '''just play the video sequence'''
        while True:
            playing = not self.paused and not self.rect_sel.dragging            
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame;

            cv2.imshow('vis', self.frame);
            ch = cv2.waitKey(1)
            if ch == 27:
                break

if __name__ == '__main__':
    print __doc__

    import sys

    use_camera = False;

    if use_camera:
        cap = video.create_capture(0);
        App(cap).run(use_camera)
    else:
        directory = '../dataset/hddd1'
        template, landmarks, imgs = distortion_util.load_dataset(directory)
        print template.shape
        cap = video.create_synth_capture("imgseq", imgs=imgs);
        App(cap, landmarks=landmarks, template=template).run(use_camera);
    #App(cap).play();
