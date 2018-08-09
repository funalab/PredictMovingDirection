import os
from glob import glob
import argparse
from shutil import copy

import numpy
from sklearn.model_selection import StratifiedKFold

cls_ref = {
    0: '0_upper_right',
    1: '1_upper_left',
    2: '2_lower_left',
    3: '3_lower_right'
}


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='./NIH3T3_annotated')
parser.add_argument('--out_dir', default='./datasets/NIH3T3_4foldcv')
parser.add_argument('--fold', '-f', type=int, default=4)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

cls_dirs = [x for x in glob(args.in_dir+'/*') if os.path.isdir(x)]
cls_dirs.sort()

fpaths = []
spaths = []
labels = []
for cls, dirpath in enumerate(cls_dirs):
    paths = glob(os.path.join(dirpath, '*.tif'))
    paths.sort()
    stats = glob(os.path.join(dirpath, '*.csv'))
    stats.sort()
    fpaths.extend(paths)
    spaths.extend(stats)
    labels.extend([cls] * len(paths))

fpaths = numpy.array(fpaths)
spaths = numpy.array(spaths)
labels = numpy.array(labels)
label_type = numpy.unique(labels)

skf = StratifiedKFold(n_splits=args.fold, shuffle=True)
fold_iter = skf.split(fpaths, labels)

fold = 0
for train_i, test_i in fold_iter:
    fold_dir = os.path.join(args.out_dir, 'fold{}'.format(fold))
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)
    train_dir = os.path.join(fold_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(fold_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for l in label_type:
        train_cls_dir = os.path.join(train_dir, cls_ref[l])
        if not os.path.exists(train_cls_dir):
            os.mkdir(train_cls_dir)
        test_cls_dir = os.path.join(test_dir, cls_ref[l])
        if not os.path.exists(test_cls_dir):
            os.mkdir(test_cls_dir)

    train_fpaths, test_fpaths = fpaths[train_i], fpaths[test_i]
    train_spaths, test_spaths = spaths[train_i], spaths[test_i]
    train_labels, test_labels = labels[train_i], labels[test_i]

    for i in range(len(train_fpaths)):
        copy(train_fpaths[i], os.path.join(train_dir, cls_ref[train_labels[i]]))
        copy(train_spaths[i], os.path.join(train_dir, cls_ref[train_labels[i]]))
    for i in range(len(test_fpaths)):
        copy(test_fpaths[i], os.path.join(test_dir, cls_ref[test_labels[i]]))
        copy(test_spaths[i], os.path.join(test_dir, cls_ref[test_labels[i]]))

    fold += 1

    with open(os.path.join(args.out_dir, 'config.txt'), 'w') as fp:
        fp.write(args.in_dir)
