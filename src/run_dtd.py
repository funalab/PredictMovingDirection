import os
import argparse
import collections as cl
import json

import numpy as np
import cv2
from chainer import cuda
from chainer import serializers
from chainer.dataset import concat_examples

from models.predict_four import predict_four
from process_dataset.handle_dataset import get_dataset
from visualize_feature import dtd as dtd
from visualize_feature.make_heatmap import cvt_rel

cls_ref = {
    0: '0_upper_right',
    1: '1_upper_left',
    2: '2_lower_left',
    3: '3_lower_right'
}

parser = argparse.ArgumentParser(description='Run layer-wise relevance propagation')
parser.add_argument('data_path', help='Input data path')
parser.add_argument('out_dir', help='Output directory path')
parser.add_argument('--param_path', default='./param.json', help='Training parameter')
parser.add_argument('--arch_path', default='./arch.json', help='Model architecture')
parser.add_argument('--model_path', default='./model.npz', help='Target model path')
parser.add_argument('--norm', '-n', type=int, default=0, help='Input-normalization mode')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
parser.add_argument('--eps', '-e', type=float, default=1e-9, help='Eps value of dtd')
parser.add_argument('--lowest', type=float, default=0.0)
parser.add_argument('--highest', type=float, default=1.0)
parser.add_argument('--colormap', '-c', default='summer')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

config_names = ['data', 'param', 'arch', 'model',
                'norm', 'gpu', 'eps', 'lowest', 'highest']
config_lst = [args.data_path, args.param_path, args.arch_path, args.model_path,
              args.norm, args.gpu, args.eps, args.lowest, args.highest]
config_dic = cl.OrderedDict()
for n, c in zip(config_names, config_lst):
    config_dic[n] = c
with open(os.path.join(args.out_dir, 'config.json'), 'w') as fp:
    json.dump(config_dic, fp)

ds_info = get_dataset(args.data_path, norm=args.norm)
data = concat_examples(ds_info.__getitem__(slice(0, ds_info.__len__(), 1)), device=args.gpu)

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
    rels, pred = dtd.dtd(model, in_array, label, eps=args.eps,
                         lowest=args.lowest, highest=args.highest)
    rels, pred = cuda.to_cpu(rels), int(pred)

    pred_dir = cor_dir if label == pred else inc_dir
    cls_dir = os.path.join(pred_dir, cls_ref[int(label)])
    fname = os.path.basename(fpath)
    fdir = os.path.join(cls_dir, os.path.splitext(fname)[0])
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    np.save(os.path.join(fdir, 'relevance.npy'), rels[pred])
    heatmap = cvt_rel(rels[pred], args.colormap)
    cv2.imwrite(os.path.join(fdir, 'rel_heatmap.tif'), heatmap)
    with open(os.path.join(fdir, 'pred_result.txt'), 'w') as fp:
        fp.write(str(pred))
