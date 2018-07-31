import collections

import numpy

from chainer import iterators
from chainer.dataset import convert


class custom_iterator(iterators.SerialIterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        super(custom_iterator, self).__init__(dataset, batch_size,
                                              repeat=repeat, shuffle=shuffle)

        label_array = convert.concat_examples(dataset)[1]
        self.labels = numpy.sort(numpy.unique(label_array))

        self.label_cnt = collections.OrderedDict()
        for l in self.labels:
            cnt = len(numpy.where(label_array == l)[0])
            self.label_cnt[l] = cnt

    def get_labels(self):
        return self.labels

    def get_label_cnt(self):
        return self.label_cnt
