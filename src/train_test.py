import os
import argparse
import collections as cl
import json

from models.predict_four import predict_four
from models.custom_classifier import custom_classifier
from optimization.trainer import train_loop
from process_dataset.handle_dataset import get_dataset
from process_dataset import proc_img

parser = argparse.ArgumentParser(description='''Train and test a CNN model for
                                                predicting the direction of
                                                cell movement''')
parser.add_argument('train_path', help='Train data path')
parser.add_argument('test_path', help='Test data path')
parser.add_argument('out_dir', help='Output directory path')
parser.add_argument('--arch_path', default='./arch.json',
                    help='Model architecutre')
parser.add_argument('--param_path', default='./param.json',
                    help='Training relevant parameter')
parser.add_argument('--norm', '-n', type=int, default=0,
                    help='Input-normalization mode')
parser.add_argument('--optname', '-o', default='MomentumSGD',
                    help='Optimizer [SGD, MomentumSGD, Adam]')
parser.add_argument('--epoch', '-e', type=int, default=50, help='Epoch number')
parser.add_argument('--test_bsize', '-b', type=int, default=32,
                    help='Batch size for test')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
parser.add_argument('--mname', '-m', default='model', help='Saved model name')
parser.add_argument('--augmodes', '-a', type=int, default=1,
                    help='Mode of data augmentation (-1: non-augmentation)')
parser.add_argument('--lr_attr', '-l', default='lr',
                    help='Learning rate attribute')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
augmodes = tuple(range(args.augmodes + 1))

train_info = get_dataset(args.train_path, norm=args.norm)
test_info = get_dataset(args.test_path, norm=args.norm)
train = train_info.__getitem__(slice(0, train_info.__len__(), 1))
test = test_info.__getitem__(slice(0, test_info.__len__(), 1))
aug_train = proc_img.augment_images(train, modes=augmodes)

with open(args.param_path) as fp:
    param = json.load(fp)
with open(args.arch_path) as fp:
    arch = json.load(fp)

predictor = predict_four(arch, dr=param['dr'], bn=param['bn'])
model = custom_classifier(predictor=predictor)

if args.mname == 'None':
    args.mname = None

trainer = train_loop()
best_score = trainer(model, aug_train, test, args.out_dir,
                     optname=args.optname, lr=param['lr'], rate=param['rate'],
                     lr_attr=args.lr_attr, gpu=args.gpu, bsize=param['bsize'],
                     test_bsize=args.test_bsize, esize=args.epoch,
                     mname=args.mname, weighting=param['weighting'],
                     l2=param['l2'])

with open(os.path.join(args.out_dir, 'best_score.json'), 'w') as fp:
    json.dump(best_score, fp)

train_dt_names = ['train', 'test', 'param', 'arch', 'norm', 'optimizer',
                  'epoch', 'test_bsize', 'gpu', 'augmodes']
train_dt_lst = [args.train_path, args.test_path, args.param_path,
                args.arch_path, args.norm, args.optname, args.epoch,
                args.test_bsize, args.gpu, augmodes]
train_dt_dic = cl.OrderedDict()

for n, c in zip(train_dt_names, train_dt_lst):
    train_dt_dic[n] = c
with open(os.path.join(args.out_dir, 'train_detail.json'), 'w') as fp:
    json.dump(train_dt_dic, fp)
