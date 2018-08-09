import os
import argparse
import pandas as pd
from glob import glob

import numpy as np
import cv2
from skimage import io

metrics = ['t_tot', 'd', 'd_tot', 'd_net', 'r_con', 'speed',
           'angle_i', 'di_ch', 'angle_net', 'v_i', 'v_cur', 'v_lin', 'r_lin']


parser = argparse.ArgumentParser(description='Annotation in moving direction')
parser.add_argument('--in_dir', default='./raw_images/U373_timelapse', help='Input data directory')
parser.add_argument('--out_dir', default='./U373_annotated', help='Output directory path')
parser.add_argument('--track', '-t', default='tracking_result.csv', help='Manual tracking result')
parser.add_argument('--interval', '-i', type=int, default=15, help='Shooting interval (min)')
parser.add_argument('--stride', '-s', type=int, default=1)
parser.add_argument('--crop', '-c', type=int, default=85)
parser.add_argument('--dst_size', '-d', type=int, default=128)
parser.add_argument('--threshold', type=float, default=18.0, help='Threshold for net displacement (um)')
parser.add_argument('--sp', type=float, default=0.65, help='Actual size per pixel (um)')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
class0_dir = os.path.join(args.out_dir, '0_upper_right')
if not os.path.exists(class0_dir):
    os.mkdir(class0_dir)
class1_dir = os.path.join(args.out_dir, '1_upper_left')
if not os.path.exists(class1_dir):
    os.mkdir(class1_dir)
class2_dir = os.path.join(args.out_dir, '2_lower_left')
if not os.path.exists(class2_dir):
    os.mkdir(class2_dir)
class3_dir = os.path.join(args.out_dir, '3_lower_right')
if not os.path.exists(class3_dir):
    os.mkdir(class3_dir)

dir_lst = [x for x in glob(os.path.join(args.in_dir, '*')) if os.path.isdir(x)]
dir_lst.sort()
sp = args.sp
for cell_dir in dir_lst:
    cell_index = os.path.basename(cell_dir.rstrip('/'))
    img_seq_path = glob(os.path.join(cell_dir, '*.tif'))
    img_seq_path.sort()
    img_seq = [io.imread(p) for p in img_seq_path]
    track = np.loadtxt(os.path.join(cell_dir, args.track),
                       delimiter=',', skiprows=1, usecols=(3, 4))
    tp = 0
    while tp < len(img_seq) and tp < len(track):
        if (track[tp][1]-args.crop < 0 or track[tp][1]+args.crop > img_seq[tp].shape[0] or
                track[tp][0]-args.crop < 0 or track[tp][0]+args.crop > img_seq[tp].shape[1]):
                tp += 1
                continue
        img_name = '{}_{}'.format(cell_index, tp)
        df = pd.DataFrame(columns=metrics)
        i = 1
        t_tot = 0
        d_tot = 0
        pre_angle_i = 0
        v_i_tot = 0
        while tp+i*args.stride < len(img_seq) and tp+i*args.stride < len(track):
            t_tot += args.interval*args.stride
            dy = sp * (track[tp+i*args.stride][1] - track[tp+i*args.stride-args.stride][1])
            dx = sp * (track[tp+i*args.stride][0] - track[tp+i*args.stride-args.stride][0])
            dy_net = sp * (track[tp+i*args.stride][1] - track[tp][1])
            dx_net = sp * (track[tp+i*args.stride][0] - track[tp][0])
            d = np.sqrt(dy**2 + dx**2)
            d_tot += d
            d_net = np.sqrt(dy_net**2 + dx_net**2)
            r_con = d_net / d_tot if not d_tot == 0 else 0
            angle_i = np.degrees(np.arctan2(dy, dx))
            di_ch = angle_i - pre_angle_i
            angle_net = np.degrees(np.arctan2(dy_net, dx_net))
            v_i = d / (args.interval*args.stride)
            v_i_tot += v_i
            v_cur = v_i_tot / i
            v_lin = d_net / t_tot
            r_lin = v_lin / v_cur if not v_cur == 0 else 0
            speed = d_tot / t_tot
            record = pd.Series([t_tot, d, d_tot, d_net, r_con, speed,
                                angle_i, di_ch, angle_net, v_i, v_cur, v_lin, r_lin],
                               index=metrics)
            df = df.append(record, ignore_index=True)
            pre_angle_i = angle_i
            i += 1
            if d_net > args.threshold:
                img = img_seq[tp][int(track[tp][1]-args.crop):int(track[tp][1]+args.crop),
                                  int(track[tp][0]-args.crop):int(track[tp][0]+args.crop)]
                img = cv2.resize(img, (args.dst_size, args.dst_size),
                                 interpolation=cv2.INTER_AREA)
                if (dy_net <= 0 and dx_net >= 0):
                    dst_dir = class0_dir
                if (dy_net <= 0 and dx_net < 0):
                    dst_dir = class1_dir
                if (dy_net > 0 and dx_net < 0):
                    dst_dir = class2_dir
                if (dy_net > 0 and dx_net >= 0):
                    dst_dir = class3_dir
                io.imsave(os.path.join(dst_dir, img_name+'.tif'), img)
                df.to_csv(os.path.join(dst_dir, img_name+'.csv'), index=False)
                break
        tp += 1
