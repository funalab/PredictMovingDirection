import six
import re
import collections

from chainer import cuda
from chainer import functions as F
from chainer import links as L

reg_conv = re.compile(r'conv_layer[0-9]')
reg_pool = re.compile(r'max_pool[0-9]')
reg_fc = re.compile(r'fc_layer[0-9]')


def guided_backprop(model, in_array, layer, top_n=5):
    xp = cuda.get_array_module(in_array)

    # Forward propagation
    if len(in_array.shape) == 3:
        in_array = in_array.reshape(1, in_array.shape[0],
                                    in_array.shape[1], in_array.shape[2])

    acts = collections.OrderedDict()
    pools = collections.OrderedDict()

    h = in_array
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

    # Guided backpropagation w.r.t each channel
    img_gbp = collections.OrderedDict()

    acts_keys = list(acts.keys())
    layer_ind = acts_keys.index(layer)

    max_acts = collections.OrderedDict()

    for ch in range(acts[layer].shape[1]):
        fmap = acts[layer][0][ch]
        max_acts[ch] = fmap.max()

    max_chs = sorted(max_acts.items(), key=lambda x: x[1], reverse=True)

    for _ch in max_chs[:top_n]:
        ch = _ch[0]
        fmap = acts[layer][0][ch]

        max_loc = fmap.argmax()
        row = int(max_loc / fmap.shape[0])
        col = int(max_loc % fmap.shape[0])

        one_hot = xp.zeros_like(acts[layer])
        one_hot[0][ch][row][col] = 1
        gx = one_hot * acts[layer]

        for key in reversed(acts_keys[:layer_ind + 1]):
            if reg_pool.match(key):
                ckey = acts_keys[acts_keys.index(key) - 1]
                gx = F.upsampling_2d(gx, pools[key].indexes, pools[key].kh,
                                     pools[key].sy, pools[key].ph,
                                     acts[ckey].shape[2:]).data

            if reg_conv.match(key):
                gx = gx * (gx > 0) * (acts[key] > 0)
                gx = F.deconvolution_2d(gx, model[key].W.data,
                                        stride=model[key].stride,
                                        pad=model[key].pad).data

        img_gbp[ch] = gx

    return img_gbp, acts['prob'][0].argmax()
