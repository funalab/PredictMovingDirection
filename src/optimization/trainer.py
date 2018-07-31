import six
import numpy

from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer.optimizer import WeightDecay
from chainer.optimizers import SGD, MomentumSGD, Adam

from optimization.testmode_evaluator import testmode_evaluator
from optimization.mca_evaluator import mca_evaluator
from optimization.scale_lr import scale_lr
from optimization.best_scoring import best_scoring
from process_dataset.custom_iterator import custom_iterator


class train_loop:
    def __init__(self):
        self.optimizers = {
            'SGD': SGD,
            'MomentumSGD': MomentumSGD,
            'Adam': Adam,
        }

    def __call__(self, model, train, test, out_dir,
                 optname='SGD', lr=1.0, rate=0.9, weighting=False,
                 gpu=-1, bsize=64, test_bsize=10, esize=50, mname=None,
                 progress=True, lr_attr='lr', l2=0.0,
                 keys=['main/loss', 'validation/main/loss', 'main/accuracy',
                       'validation/main/accuracy', 'validation_in_mca/main/mca'],
                 s_keys=['validation_in_mca/main/mca'],
                 p_keys=['epoch', 'main/loss', 'validation/main/loss',
                         'main/accuracy', 'validation/main/accuracy',
                         'validation_in_mca/main/mca', 'elapsed_time']):

        train_iter = custom_iterator(train, batch_size=bsize)
        test_iter = custom_iterator(test, batch_size=test_bsize,
                                    repeat=False, shuffle=False)

        if weighting:
            label_cnt = train_iter.get_label_cnt()
            n_cls = len(label_cnt.keys())
            cls_weight = numpy.empty(n_cls)
            for k, cnt in six.iteritems(label_cnt):
                cls_weight[k] = cnt
            cls_weight = (cls_weight.sum() / cls_weight / n_cls).astype(numpy.float32)
        else:
            cls_weight = None
        if gpu >= 0:
            cuda.get_device(gpu).use()
            model.to_gpu()
            if cls_weight is not None:
                cls_weight = cuda.to_gpu(cls_weight)
        model.cls_weight = cls_weight

        optimizer = self.optimizers[optname](lr)
        optimizer.setup(model)
        if l2 > 0:
            optimizer.add_hook(WeightDecay(l2))

        updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

        trainer = training.Trainer(updater, (esize, 'epoch'), out=out_dir)
        trainer.extend(testmode_evaluator(test_iter, model, device=gpu))
        trainer.extend(mca_evaluator(test_iter, model, device=gpu))
        trainer.extend(extensions.LogReport())
        trainer.extend(scale_lr(attr=lr_attr, rate=rate))
        trainer.extend(best_scoring(model, keys, s_keys=s_keys, mname=mname))
        if progress:
            trainer.extend(extensions.PrintReport(p_keys))
            trainer.extend(extensions.ProgressBar())
        trainer.run()
        return model.predictor.best_score


if __name__ == '__main__':
    import argparse

    from models.custom_classifier import custom_classifier
    from process_dataset.handle_dataset import get_dataset
    from models.predict_four import predict_four

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--weighting', '-w', type=bool, default=False)
    parser.add_argument('--mname', '-m', default=None)
    args = parser.parse_args()

    train_info = get_dataset('test_data/test_dataset/train', norm=1)
    test_info = get_dataset('test_data/test_dataset/test', norm=1)
    train = train_info.__getitem__(slice(0, train_info.__len__(), 1))
    test = test_info.__getitem__(slice(0, test_info.__len__(), 1))
    predictor = predict_four()
    model = custom_classifier(predictor)
    loop = train_loop()
    max_acc, trainer = loop(model, train, test, 'test_data/results',
                            bsize=10, esize=5, gpu=args.gpu,
                            weighting=args.weighting, mname=args.mname)
