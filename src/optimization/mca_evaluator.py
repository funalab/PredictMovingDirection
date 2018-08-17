import copy

import six

from chainer.training import extension
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer import variable
from chainer import cuda


class mca_evaluator(extension.Extension):
    trigger = 1, 'epoch'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, name='validation_in_mca'):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.name = name

    def get_iterator(self, name):
        return self._iterators[name]

    def get_all_iterators(self):
        return dict(self._targets)

    def get_target(self, name):
        return self._targets[name]

    def get_all_targets(self):
        return dict(self._targets)

    def __call__(self, trainer=None):
        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''

        mca = self.evaluate()
        result = {prefix + 'main/mca': mca}

        reporter_module.report(result)
        return result

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        dic_summary = reporter_module.DictSummary()
        labels = it.get_labels()

        target.predictor.train = False
        for batch in it:
            observation = {}
            in_arrays = self.converter(batch, self.device)
            if isinstance(in_arrays, tuple):
                in_vars = tuple(variable.Variable(x, volatile='on')
                                for x in in_arrays)

                xp = cuda.get_array_module(*in_vars)
                y = target.predictor(in_vars[0]).data
                t = in_vars[1].data
                pred = y.argmax(axis=1).reshape(t.shape)

                for l in labels:
                    ind = xp.where(t == l)[0]
                    if len(ind) == 0:
                        t_cnt = 0
                    else:
                        t_cnt = len(xp.where(pred[ind] == l)[0])
                    observation.update({l: t_cnt})

            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x, volatile='on')
                           for key, x in six.iteritems(in_arrays)}
                pred = target.predictor(**in_vars)
                # TODO
            else:
                in_var = variable.Variable(in_arrays, volatile='on')
                pred = target.predictor(in_var)
                # TODO
            dic_summary.add(observation)

        label_cnt = it.get_label_cnt()
        mca = xp.array([float(summary._x) / float(label_cnt[l]) for l, summary
                        in six.iteritems(dic_summary._summaries)]).mean(dtype=y.dtype)
        target.predictor.train = True
        return mca


if __name__ == '__main__':
    from chainer import links as L
    from process_dataset.handle_dataset import get_dataset
    from models.predict_four import predict_four
    from process_dataset.custom_iterator import custom_iterator

    test_info = get_dataset('test_data/test_dataset/test', norm=1)
    test = test_info.__getitem__(slice(0, test_info.__len__(), 1))
    test_iter = custom_iterator(test, batch_size=10, repeat=False, shuffle=False)
    predictor = predict_four()
    model = L.Classifier(predictor)
    evaluator = mca_evaluator(test_iter, model)
    mca = evaluator.evaluate()
