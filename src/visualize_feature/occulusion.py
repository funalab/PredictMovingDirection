import copy
import six
import numpy as np
import skimage.io as io
from chainer.training import extension
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer import variable
from chainer import cuda
from optimization.mca_evaluator import mca_evaluator
from visualize_feature.make_heatmap import cvt_rel

def occulusion(target, in_array, label, kernel, gpu):
    hk = int(kernel / 2)
    occulusion_map = np.zeros((128, 128)).astype(np.float16)
    for num_x in range(128):
        for num_y in range(128):
            if gpu >= 0:
                in_array = cuda.to_cpu(in_array)
            in_pad = copy.deepcopy(
                    np.pad(in_array[0], pad_width=hk, mode='reflect')
                )
            in_pad[num_y:num_y+kernel, num_x:num_x+kernel] = 0
            in_pad = copy.deepcopy(in_pad[hk:-hk, hk:-hk])
            if gpu >= 0:
                in_pad = cuda.to_gpu(np.expand_dims(np.expand_dims(in_pad, 0), 0))
            else:
                in_pad = np.expand_dims(np.expand_dims(in_pad, 0), 0)
            y = target.predictor(in_pad).data
            pred = y.argmax(axis=1).reshape(label.shape)
            occulusion_map[num_y][num_x] += y[0][label]
    return occulusion_map, pred
