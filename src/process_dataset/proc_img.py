# -*- coding: utf-8 -*-

import numpy as np


def augment_images(data, modes=(0, 1)):
    if len(modes) == 0:
        return data
    aug_data = []
    if 0 in modes:
        for img, label in data:
            rot_data = rotate_four(img, label)
            aug_data.extend(rot_data)
            if 1 in modes:
                flip_data = flip_four(img, label)
                aug_data.extend(flip_data)
    return aug_data


def rotate_four(img, label):
    rot0 = img
    if len(img.shape) == 3:
        img = img.reshape(img.shape[1], img.shape[2])
    rot90 = np.rot90(img).reshape(1, img.shape[0], img.shape[1])
    rot180 = np.rot90(img, 2).reshape(1, img.shape[0], img.shape[1])
    rot270 = np.rot90(img, 3).reshape(1, img.shape[0], img.shape[1])
    rot_labels = [l if l <= 3 else l-4 for l in [label, label+1, label+2, label+3]]
    rot_data = [(rot0, np.int32(rot_labels[0])), (rot90, np.int32(rot_labels[1])),
                (rot180, np.int32(rot_labels[2])), (rot270, np.int32(rot_labels[3]))]
    return rot_data


def flip_four(img, label):
    if len(img.shape) == 3:
        img = img.reshape(img.shape[1], img.shape[2])
    hor = np.flip(img, 0).reshape(1, img.shape[0], img.shape[1])
    ver = np.flip(img, 1).reshape(1, img.shape[0], img.shape[1])
    label_hor = abs(label - 3)
    label_ver = label + 1 if label == 2 else abs(label - 1)
    flip_data = [(hor, np.int32(label_hor)), (ver, np.int32(label_ver))]
    return flip_data


if __name__ == '__main__':
    from process_dataset.handle_dataset import get_dataset
    train_info = get_dataset('test_data/test_dataset/train', norm=1)
    train = train_info.__getitem__(slice(0, train_info.__len__(), 1))
    train_aug = augment_images(train)
