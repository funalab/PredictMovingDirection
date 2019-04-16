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

def occulusion(target, in_array, label, kernel):
    hk = int(kernel / 2)
    occulusion_map = np.zeros((128, 128)).astype(np.float16)
    for num_x in range(128):
        for num_y in range(128):
            in_pad = copy.deepcopy(
                    np.pad(in_array[0], pad_width=hk, mode='reflect')
                )
            in_pad[num_y:num_y+kernel, num_x:num_x+kernel] = 0
            in_pad = copy.deepcopy(in_pad[hk:-hk, hk:-hk])
            y = target.predictor(np.expand_dims(np.expand_dims(in_pad, 0), 0)).data
            pred = y.argmax(axis=1).reshape(label.shape)
            occulusion_map[num_y][num_x] += y[0][label]
    return occulusion_map, pred
