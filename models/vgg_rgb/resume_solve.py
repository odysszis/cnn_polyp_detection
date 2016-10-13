import caffe
import surgery, score
import numpy as np
import os
import sys 

# set gpu mode
caffe.set_mode_gpu()
caffe.set_device(1);

# Initialize SGD solver and restore the snapshot state
solver = caffe.SGDSolver('solver.prototxt')
solver.restore('snapshot/train_iter_50000.solverstate')

# Resize blobs corresponding to deconvolutions
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# Resume training
solver.solve()
