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

def occulusion(target, in_array, label):
    kernel = 7
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


class occulusion_mca_evaluator(mca_evaluator):
    def evaluate(self, out_dir):
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

        self.kernel = 7
        hk = int(self.kernel / 2)

        for batch in it:
            observation = {}
            in_arrays = self.converter(batch, self.device)
            occulusion_map = np.zeros((128, 128)).astype(np.float16)
            for num_x in range(128):
                for num_y in range(128):
                    if isinstance(in_arrays, tuple):
                        #in_vars = tuple(variable.Variable(x, volatile='on')
                        in_vars = tuple(variable.Variable(x)
                                        for x in in_arrays)

                        xp = cuda.get_array_module(*in_vars)
                        #print(np.shape(in_vars[0]))
                        in_pad = copy.deepcopy(
                                np.pad(in_vars[0].data[0][0], pad_width=hk, mode='reflect')
                            )
                        #print(np.shape(in_pad))
                        in_pad[num_y:num_y+self.kernel, num_x:num_x+self.kernel] = 0
                        in_pad = copy.deepcopy(in_pad[hk:-hk, hk:-hk])
                        #print(np.shape(in_vars[0]))
                        #print(in_pad[:5, :5])
                        y = target.predictor(np.expand_dims(np.expand_dims(in_pad, 0), 0)).data
                        t = in_vars[1].data
                        pred = y.argmax(axis=1).reshape(t.shape)
                    occulusion_map[num_y][num_x] += y[0][t]

            heatmap = cvt_rel(occulusion_map, args.colormap)
            io.imsave(os.path.join(out_dir, ), occulusion_map)
            sys.exit()

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

    evaluator = occulusion_mca_evaluator(test_iter, model)
    mca_cls, count_summary = evaluator.evaluate(args.out_dir)
    mca_mean = np.mean(mca_cls)
    mca_std = np.std(mca_cls)

    with open(os.path.join(args.out_dir, 'test_score.txt'), 'w') as fp:
        fp.write('mca_cls: {}\n'.format(mca_cls))
        fp.write('mca_mean: {}\n'.format(mca_mean))
        fp.write('mca_std: {}\n'.format(mca_std))
    with open(os.path.join(args.out_dir, 'confusion_matrix.txt'), 'w') as fp:
        fp.write('upper_right: {}\n'.format(count_summary[0]))
        fp.write('upper_left: {}\n'.format(count_summary[1]))
        fp.write('lower_left: {}\n'.format(count_summary[2]))
        fp.write('lower_right: {}\n'.format(count_summary[3]))
