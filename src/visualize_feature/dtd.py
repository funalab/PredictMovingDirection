import collections
import six
import re

from chainer import cuda
from chainer import functions as F
from chainer import links as L

reg_conv = re.compile(r'conv_layer[0-9]')
reg_pool = re.compile(r'max_pool[0-9]')
reg_fc = re.compile(r'fc_layer[0-9]')


def dtd(model, in_array, label, eps=1e-9, lowest=0.0, highest=1.0):
    xp = cuda.get_array_module(in_array)

    # Forward propagation
    if len(in_array.shape) == 3:
        in_array = in_array.reshape(1, in_array.shape[0],
                                    in_array.shape[1], in_array.shape[2])

    acts = collections.OrderedDict()
    pools = collections.OrderedDict()

    h = in_array
    acts['input'] = in_array
    for key, funcs in six.iteritems(model.functions):
        for func in funcs:
            if isinstance(func, L.BatchNormalization):
                h = func(h, test=not model.train)

            elif func is F.dropout:
                h = func(h, ratio=model.dr, train=model.train)

            elif isinstance(func, F.MaxPooling2D):
                func.use_cudnn = False
                pools[key] = func
                h = func(h)
            else:
                h = func(h)

        acts[key] = h.data

    # Layer-wise propagation by deep taylor decomposition
    img_rel = xp.empty((acts['prob'].shape[1], in_array.shape[0],
                        in_array.shape[1], in_array.shape[2],
                        in_array.shape[3]))

    acts_keys = list(acts.keys())

    for cls in range(acts['prob'].shape[1]):
        for key in reversed(acts):
            if key == 'prob':
                continue

            pre_key = acts_keys[acts_keys.index(key) - 1]

            if key == 'out_layer':
                v = xp.maximum(0, model['out_layer'].W.data)
                z = xp.dot(v[cls], acts[pre_key][0])
                s = acts['prob'][0][cls] / z
                r = v[cls] * acts[pre_key][0] * s

            if reg_fc.match(key):
                if reg_pool.match(pre_key):
                    pre_act = acts[pre_key].reshape(acts[pre_key].size)
                else:
                    pre_act = acts[pre_key]

                v = xp.maximum(0, model[key].W.data)
                z = xp.dot(v, pre_act)
                s = r / (z + xp.copysign(eps, z))
                c = xp.dot(s, v)
                r = pre_act * c

            if reg_pool.match(key):
                if r.ndim == 1:
                    r = r.reshape(acts[key].shape)
                r = F.upsampling_2d(r, pools[key].indexes, pools[key].kh,
                                    pools[key].sy, pools[key].ph,
                                    acts[pre_key].shape[2:]).data

            if reg_conv.match(key):
                v = xp.maximum(0, model[key].W.data)

                if pre_key == 'input':
                    w = model[key].W.data
                    u = xp.minimum(0, model[key].W.data)

                    l_map = (xp.ones_like(acts['input']) * lowest).astype('f')
                    h_map = (xp.ones_like(acts['input']) * highest).astype('f')

                    z = F.convolution_2d(acts['input'], w).data -\
                        F.convolution_2d(l_map, v).data -\
                        F.convolution_2d(h_map, u).data

                    s = r / (z + xp.copysign(eps, z))

                    r = acts['input'] * F.deconvolution_2d(s, w).data -\
                        l_map * F.deconvolution_2d(s, v).data - \
                        h_map * F.deconvolution_2d(s, u).data

                    break

                else:
                    z = F.convolution_2d(acts[pre_key], v).data
                    s = r / (z + xp.copysign(eps, z))
                    r = acts[pre_key] * F.deconvolution_2d(s, v).data

        img_rel[cls] = r

    return img_rel, acts['prob'].argmax()
