import os
import argparse
import collections as cl
import json

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from chainer import cuda
from chainer import serializers
from chainer.dataset import concat_examples

from models.predict_four import predict_four
from process_dataset.handle_dataset import get_dataset
from visualize_feature.gbp import guided_backprop
from visualize_feature import convert as C

cls_ref = {
    0: '0_upper_right',
    1: '1_upper_left',
    2: '2_lower_left',
    3: '3_lower_right'
}

parser = argparse.ArgumentParser(description='Run guided grad-gcam')
parser.add_argument('data_path', help='Input data path')
parser.add_argument('out_dir', help='Output directory path')
parser.add_argument('--param_path', default='./param.json', help='Training parameter')
parser.add_argument('--arch_path', default='./arch.json', help='Model architecture')
parser.add_argument('--model_path', default='./model.npz', help='Target model path')
parser.add_argument('--norm', '-n', type=int, default=0,
                    help='Input-normalization mode')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
parser.add_argument('--layer', '-l', default='conv_layer7',
                    help='Target conv layer for grad cam')
parser.add_argument('--top_n', '-t', type=int, default=3)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

config_names = ['data', 'param', 'arch', 'model', 'norm', 'gpu', 'layer']
config_lst = [args.data_path, args.param_path, args.arch_path, args.model_path,
              args.norm, args.gpu, args.layer]
config_dic = cl.OrderedDict()
for n, c in zip(config_names, config_lst):
    config_dic[n] = c
with open(os.path.join(args.out_dir, 'config.json'), 'w') as fp:
    json.dump(config_dic, fp)

ds_info = get_dataset(args.data_path, norm=args.norm)
data = concat_examples(ds_info.__getitem__(slice(0, ds_info.__len__(), 1)),
                       device=args.gpu)

with open(args.param_path, 'r') as fp:
    param = json.load(fp)
with open(args.arch_path, 'r') as fp:
    arch = json.load(fp)

model = predict_four(arch, dr=param['dr'], bn=param['bn'])
serializers.load_npz(args.model_path, model)
if args.gpu >= 0:
    cuda.get_device(data[0][0]).use()
    model.to_gpu()
    xp = cuda.get_array_module(data[0][0])
model.train = False

cor_dir = os.path.join(args.out_dir, 'correct')
if not os.path.exists(cor_dir):
    os.mkdir(cor_dir)
inc_dir = os.path.join(args.out_dir, 'incorrect')
if not os.path.exists(inc_dir):
    os.mkdir(inc_dir)
for cls_dir in ds_info.class_dirs:
    cls = os.path.basename(cls_dir)
    cor_cls_dir = os.path.join(cor_dir, cls)
    if not os.path.exists(cor_cls_dir):
        os.mkdir(cor_cls_dir)
    inc_cls_dir = os.path.join(inc_dir, cls)
    if not os.path.exists(inc_cls_dir):
        os.mkdir(inc_cls_dir)

for i in range(ds_info.__len__()):
    in_array, label, fpath = data[0][i], data[1][i], ds_info.data_pairs[i][0]

    img = io.imread(fpath)
    if img.ndim >= 3:
        img = rgb2gray(img)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    gbps, pred = guided_backprop(model, in_array, args.layer, args.top_n)

    pred_dir = cor_dir if label == pred else inc_dir
    cls_dir = os.path.join(pred_dir, cls_ref[int(label)])
    fname = os.path.basename(fpath)
    fdir = os.path.join(cls_dir, os.path.splitext(fname)[0])
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    rank = 0
    for ch, gbp in gbps.items():
        gbp = C.convert_gbp(gbp, img.shape)
        io.imsave(os.path.join(fdir, 'local_feature_rank{}_ch{}.tif'.format(rank, ch)),
                  gbp.astype(np.uint8))
        rank += 1
    with open(os.path.join(fdir, 'pred_result.txt'), 'w') as fp:
        fp.write(str(pred))
