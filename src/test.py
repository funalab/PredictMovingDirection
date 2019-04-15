import copy
import six
from chainer.training import extension
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer import variable
from chainer import cuda
from optimization.mca_evaluator import mca_evaluator

class test_mca_evaluator(mca_evaluator):
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

        count_summary = {
            0:[0,0,0,0],
            1:[0,0,0,0],
            2:[0,0,0,0],
            3:[0,0,0,0]
        }
        target.predictor.train = False
        for batch in it:
            observation = {}
            in_arrays = self.converter(batch, self.device)
            if isinstance(in_arrays, tuple):
                #in_vars = tuple(variable.Variable(x, volatile='on')
                in_vars = tuple(variable.Variable(x)
                                for x in in_arrays)

                xp = cuda.get_array_module(*in_vars)
                y = target.predictor(in_vars[0]).data
                t = in_vars[1].data
                pred = y.argmax(axis=1).reshape(t.shape)
                count_summary[t[0]][pred[0]] += 1

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
                #in_var = variable.Variable(in_arrays, volatile='on')
                in_var = variable.Variable(in_arrays)
                pred = target.predictor(in_var)
                # TODO
            dic_summary.add(observation)

        label_cnt = it.get_label_cnt()
        mca_class = xp.array([float(summary._x) / float(label_cnt[l]) for l, summary
                        in six.iteritems(dic_summary._summaries)]).astype(dtype=y.dtype)
        target.predictor.train = True
        return mca_class, count_summary


if __name__ == '__main__':
    import os
    import argparse
    import collections as cl
    import json
    import numpy as np
    from chainer import serializers

    from models.predict_four import predict_four
    from models.custom_classifier import custom_classifier
    from optimization.trainer import train_loop
    from process_dataset.handle_dataset import get_dataset
    from process_dataset import proc_img
    from process_dataset.custom_iterator import custom_iterator

    parser = argparse.ArgumentParser(description='''Test a CNN model for predicting
                                                    the direction of cell movement''')
    parser.add_argument('test_path', help='Test data path')
    parser.add_argument('out_dir', help='Output directory path')
    parser.add_argument('--arch_path', default='./arch.json',
                        help='Model architecutre')
    parser.add_argument('--param_path', default='./param.json',
                        help='Training relevant parameter')
    parser.add_argument('--norm', '-n', type=int, default=0,
                        help='Input-normalization mode')
    parser.add_argument('--test_bsize', '-b', type=int, default=1,
                        help='Batch size for test')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
    parser.add_argument('--model', '-m', default='model.npz', help='Load model path')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    test_info = get_dataset(args.test_path, norm=args.norm)
    test = test_info.__getitem__(slice(0, test_info.__len__(), 1))
    test_iter = custom_iterator(test, batch_size=args.test_bsize, repeat=False, shuffle=False)

    with open(args.param_path) as fp:
        param = json.load(fp)
    with open(args.arch_path) as fp:
        arch = json.load(fp)

    predictor = predict_four(arch, dr=param['dr'], bn=param['bn'])
    if args.model == 'None':
        args.model = None
    else:
        serializers.load_npz(args.model, predictor)
        print('Initialization model: {}'.format(args.model))
    model = custom_classifier(predictor=predictor)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    evaluator = test_mca_evaluator(test_iter, model)
    mca_cls, count_summary = evaluator.evaluate()
    mca_mean = np.mean(mca_cls)
    mca_std = np.std(mca_cls)
    test_dt_dic = cl.OrderedDict()
    cls_name = ['0', '1', '2', '3']

    for n, c in zip(cls_name, mca_cls):
        test_dt_dic[n] = c

    with open(os.path.join(args.out_dir, 'test_score.txt'), 'w') as fp:
        fp.write('mca_cls: {}\n'.format(mca_cls))
        fp.write('mca_mean: {}\n'.format(mca_mean))
        fp.write('mca_std: {}\n'.format(mca_std))
    with open(os.path.join(args.out_dir, 'confusion_matrix.txt'), 'w') as fp:
        fp.write('upper_right: {}\n'.format(count_summary[0]))
        fp.write('upper_left: {}\n'.format(count_summary[1]))
        fp.write('lower_left: {}\n'.format(count_summary[2]))
        fp.write('lower_right: {}\n'.format(count_summary[3]))
