import caffe
import surgery, score
import pdb
import numpy as np
import os, sys
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# set gpu mode
caffe.set_mode_gpu()
caffe.set_device(0);

# RGB VGG architecture prototxt path
prototxt_file = 'fcn32s.prototxt'
# Pre-trained weights path of RGB model
weights = 'fcn32s-heavy-pascal.caffemodel'

# Initialize RGB model to copy 3 input filter weights (corresponding to RGB)
base_net = caffe.Net(prototxt_file, weights, caffe.TRAIN)

# Initialize SGD solver for the RGBD CNN
solver = caffe.SGDSolver('solver.prototxt')

# copy filter weights from the RGB model to the RGBD model
# this will copy weights from the parameters with the same
# name in the RGB and RGBD model. Since the input layer will
# be 4-channel instead of 3-channel (RGBD instead of RGB), it
# has a different name, so the weights will not be copied
surgery.transplant(solver.net, base_net)

# Resize blobs corresponding to deconvolutions
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# Copy the filters of RGB input to the first 3 filters of the RGBD CNN input
solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params['conv1_1'][0].data
# Initialize the depth channel filter weights with the average of the RGB weights
solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
# Copy the filter bias terms
solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params['conv1_1'][1].data

del base_net

# Start training
solver.solve()
