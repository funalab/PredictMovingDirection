import os
import argparse
from glob import glob
import collections as cl
import json

import numpy as np

import cv2

parser = argparse.ArgumentParser(description='Visualize pixel-wise relevance')
parser.add_argument('data_path', help='Input data path')
parser.add_argument('out_dir', help='Output directory path')
parser.add_argument('--stat', '-s', default=None, help='Relevance-Statistics file path')
parser.add_argument('--make_map', '-m', default=True, help='Whether make heatmap')
parser.add_argument('--norm', '-n', default='imgwise')
parser.add_argument('--colormap', '-c', default='summer')
args = parser.parse_args()

colormaps = {
    'jet': cv2.COLORMAP_JET,
    'summer': cv2.COLORMAP_SUMMER
}

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

if args.stat is None:
    rels = []
    correct_dir = os.path.join(args.data_path, 'correct')
    failed_dir = os.path.join(args.data_path, 'failed')
    ccls_dir = [x for x in glob(correct_dir+'/*') if os.path.isdir(x)]
    ccls_dir.sort()
    fcls_dir = [x for x in glob(failed_dir+'/*') if os.path.isdir(x)]
    fcls_dir.sort()
    for ccls, fcls in zip(ccls_dir, fcls_dir):
        ccls_cells = [x for x in glob(ccls+'/*') if os.path.isdir(x)]
        fcls_cells = [x for x in glob(fcls+'/*') if os.path.isdir(x)]
        for cell in ccls_cells:
            rel_paths = glob(cell+'/*.npy')
            rels.extend(map(np.load, rel_paths))
        for cell in fcls_cells:
            rel_paths = glob(cell+'/*.npy')
            rels.extend(map(np.load, rel_paths))
    rels = np.array(rels)
    met_names = ['max', 'min', 'mean', 'N']
    met_lst = [rels.max(), rels.min(), rels.mean(), rels.shape[0]]
    met_dic = cl.OrderedDict()
    for n, c in zip(met_names, met_lst):
        met_dic[n] = c
    with open(os.path.join(args.data_path, 'stat.json'), 'w') as fp:
        json.dump(met_dic, fp)
    gmax_rel = met_dic['max']
    gmin_rel = met_dic['min']
else:
    with open(args.stat, 'r') as fp:
        stat = json.load(fp)
    gmax_rel = stat['max']
    gmin_rel = stat['min']

if args.make_map:
    correct_dir = os.path.join(args.data_path, 'correct')
    failed_dir = os.path.join(args.data_path, 'failed')
    ccls_dir = [x for x in glob(correct_dir+'/*') if os.path.isdir(x)]
    ccls_dir.sort()
    fcls_dir = [x for x in glob(failed_dir+'/*') if os.path.isdir(x)]
    fcls_dir.sort()
    out_correct_dir = os.path.join(args.out_dir, 'correct')
    if not os.path.exists(out_correct_dir):
        os.mkdir(out_correct_dir)
    out_failed_dir = os.path.join(args.out_dir, 'failed')
    if not os.path.exists(out_failed_dir):
        os.mkdir(out_failed_dir)
    for cls in range(len(ccls_dir)):
        out_ccls_dir = os.path.join(out_correct_dir, 'class{}'.format(cls))
        if not os.path.exists(out_ccls_dir):
            os.mkdir(out_ccls_dir)
        out_fcls_dir = os.path.join(out_failed_dir, 'class{}'.format(cls))
        if not os.path.exists(out_fcls_dir):
            os.mkdir(out_fcls_dir)
    cls = 0
    for ccls, fcls in zip(ccls_dir, fcls_dir):
        ccls_cells = [x for x in glob(ccls+'/*') if os.path.isdir(x)]
        fcls_cells = [x for x in glob(fcls+'/*') if os.path.isdir(x)]
        for cell in ccls_cells:
            out_cell_dir = os.path.join(out_correct_dir, 'class{}'.format(cls), os.path.basename(cell))
            if not os.path.exists(out_cell_dir):
                os.mkdir(out_cell_dir)
            rel_paths = glob(cell+'/*.npy')
            cell_rels = []
            for rel_path in rel_paths:
                rel = np.load(rel_path)
                rel = rel.reshape(rel.shape[2], rel.shape[3])
                cell_rels.append(rel)
            cell_rels = np.array(cell_rels)
            cmax_rel = cell_rels.max()
            cmin_rel = cell_rels.min()
            for rel_path in rel_paths:
                rel = np.load(rel_path)
                rel = rel.reshape(rel.shape[2], rel.shape[3])
                if args.norm == 'global':
                    max_rel, min_rel = gmax_rel, gmin_rel
                elif args.norm == 'class':
                    max_rel, min_rel = cmax_rel, cmin_rel
                else:
                    max_rel, min_rel = rel.max(), rel.min()
                rel = (rel - min_rel) / (max_rel - min_rel) * 255
                rel = rel.astype(np.uint8)
                rel_map = cv2.applyColorMap(rel, colormaps[args.colormap])
                rel_name = os.path.basename(rel_path)
                map_name = os.path.splitext(rel_name)[0] + '.tif'
                save_path = os.path.join(out_cell_dir, map_name)
                cv2.imwrite(save_path, rel_map)
        for cell in fcls_cells:
            out_cell_dir = os.path.join(out_failed_dir, 'class{}'.format(cls), os.path.basename(cell))
            if not os.path.exists(out_cell_dir):
                os.mkdir(out_cell_dir)
            rel_paths = glob(cell+'/*.npy')
            cell_rels = []
            for rel_path in rel_paths:
                rel = np.load(rel_path)
                rel = rel.reshape(rel.shape[2], rel.shape[3])
                cell_rels.append(rel)
            cell_rels = np.array(cell_rels)
            cmax_rel = cell_rels.max()
            cmin_rel = cell_rels.min()
            for rel_path in rel_paths:
                rel = np.load(rel_path)
                rel = rel.reshape(rel.shape[2], rel.shape[3])
                if args.norm == 'global':
                    max_rel, min_rel = gmax_rel, gmin_rel
                elif args.norm == 'class':
                    max_rel, min_rel = cmax_rel, cmin_rel
                else:
                    max_rel, min_rel = rel.max(), rel.min()
                rel = (rel - min_rel) / (max_rel - min_rel) * 255
                rel = rel.astype(np.uint8)
                rel_map = cv2.applyColorMap(rel, colormaps[args.colormap])
                rel_name = os.path.basename(rel_path)
                map_name = os.path.splitext(rel_name)[0] + '.tif'
                save_path = os.path.join(out_cell_dir, map_name)
                cv2.imwrite(save_path, rel_map)
        cls += 1

config_names = ['data', 'norm', 'colormap']
config_lst = [args.data_path, args.norm, args.colormap]
config_dic = cl.OrderedDict()
for n, c in zip(config_names, config_lst):
    config_dic[n] = c
with open(os.path.join(args.out_dir, 'config.json'), 'w') as fp:
    json.dump(config_dic, fp)
