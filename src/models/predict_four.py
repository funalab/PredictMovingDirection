import collections
import re
import six

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer import Variable

reg_conv = re.compile(r'conv_layer[0-9]')
reg_pool = re.compile(r'max_pool[0-9]')
reg_fc = re.compile(r'fc_layer[0-9]')
reg_num = re.compile(r'[0-9]+')


class predict_four(Chain):
    def __init__(self, arch, dr=0.5, bn=False):
        super(predict_four, self).__init__()

        self.dr = dr
        self.train = True
        self.best_score = collections.defaultdict(lambda: 0)
        self.functions = collections.OrderedDict()

        for l in arch['0']:
            name = l['name']

            if reg_conv.match(name):
                self.add_link(name,
                              L.Convolution2D(None, l['out_channel'],
                                              l['ksize'], pad=l['pad']))
                if bn:
                    m = reg_num.search(name)
                    bn_name = 'cbn_layer{}'.format(m.group())
                    self.add_link(bn_name,
                                  L.BatchNormalization(l['out_channel']))
                    self.functions[name] = [self[name], self[bn_name],
                                            F.relu]
                else:
                    self.functions[name] = [self[name], F.relu]

            if reg_pool.match(name):
                self.functions[name] = [F.MaxPooling2D(l['ksize'], l['stride'])]

            if reg_fc.match(name):
                self.add_link(name,
                              L.Linear(l['in_size'], l['out_size']))
                if bn:
                    m = reg_num.search(name)
                    bn_name = 'fbn_layer{}'.format(m.group())
                    self.add_link(bn_name,
                                  L.BatchNormalization(l['out_size']))
                    self.functions[name] = [self[name], self[bn_name],
                                            F.relu, F.dropout]
                else:
                    self.functions[name] = [self[name], F.relu, F.dropout]

            if name == 'out_layer':
                self.add_link(name,
                              L.Linear(None, l['out_size']))
                self.functions[name] = [self[name]]

        self.functions['prob'] = [F.softmax]

    def __call__(self, x):
        with chainer.using_config('train', self.train):
            h = x
            for key, funcs in six.iteritems(self.functions):
                if key == 'prob':
                    break
                for func in funcs:
                    if isinstance(func, L.BatchNormalization):
                        h = func(h, test=not self.train)
                    elif func is F.dropout:
                        #h = func(h, ratio=self.dr, train=self.train)
                        h = func(h, ratio=self.dr)
                    else:
                        h = func(h)
        return h

    def extract(self, x, layers=['prob']):
        if len(x.shape) == 3:
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        if not isinstance(x, Variable):
            x = Variable(x)

        h = x
        activations = {'input': x}
        target_layers = set(layers)

        for key, funcs in six.iteritems(self.functions):
            if len(target_layers) == 0:
                break
            for func in funcs:
                if isinstance(func, L.BatchNormalization):
                    h = func(h, test=not self.train)
                elif func is F.dropout:
                    h = func(h, ratio=self.dr, train=self.train)
                elif func is F.MaxPooling2D:
                    func.use_cudnn = False
                    h = func(h)
                else:
                    h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)

        return activations


if __name__ == '__main__':
    import argparse
    import json

    import numpy as np
    from skimage import io

    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    args = parser.parse_args()

    row_img1 = io.imread('test_data/NIH_1012_1_0.tif')
    row_img2 = io.imread('test_data/NIH_1016_4_1.tif')

    img1 = (row_img1 - row_img1.min())/(row_img1.max() - row_img1.min())
    img1 = img1.reshape(1, row_img1.shape[0], row_img1.shape[1])
    img2 = (row_img2 - row_img2.min())/(row_img2.max() - row_img2.min())
    img2 = img2.reshape(1, row_img2.shape[0], row_img2.shape[1])

    in_array = np.asarray([img1, img2]).astype(np.float32)

    with open(args.arch, 'r') as fp:
        arch = json.load(fp)

    model = predict_four(arch=arch)
    model.train = False

    y = model(in_array)
    act = model.extract(in_array, layers=['conv_layer6', 'out_layer', 'prob'])
