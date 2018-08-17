from chainer import cuda


def convert_gbp(gbp, shape):
    gbp = gbp.reshape(shape)
    gbp -= gbp.min()
    gbp = 255 * gbp / gbp.max()

    return cuda.to_cpu(gbp)
