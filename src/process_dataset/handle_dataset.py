import os
from glob import glob
import numpy as np
from skimage import io

import chainer


class get_dataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_path, norm=0, dtype=np.float32):
        self.class_dirs = [x for x in glob(data_path+'/*') if os.path.isdir(x)]
        self.class_dirs.sort()
        self.data_pairs = []
        self.file_order = []
        for cls_index, cls_path in enumerate(self.class_dirs):
            paths = glob(os.path.join(cls_path, '*.tif'))
            paths.sort()
            for file_path in paths:
                self.data_pairs.append((file_path, cls_index))
                file_name = os.path.basename(file_path)
                file_index, _ = os.path.splitext(file_name)
                self.file_order.append(file_index)
        self.norm = norm
        self.dtype = dtype
        self.norm_method = {
            0: self.bright_norm,
            1: self.eightbit_norm,
            2: self.scale_norm,
        }

    def __len__(self):
        return len(self.data_pairs)

    def get_example(self, i):
        path, label = self.data_pairs[i]
        img = io.imread(path).astype(self.dtype)
        img = self.norm_method[self.norm](img)
        img = img.reshape(1, img.shape[0], img.shape[1])
        return img, np.int32(label)

    def bright_norm(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def eightbit_norm(self, img):
        return ((((img-img.min())*255/(img.max()-img.min())).astype(np.uint8))/255.0).astype(self.dtype)

    def scale_norm(self, img):
        return img / 255.0


if __name__ == '__main__':
    ds_info = get_dataset('test_data/test_dataset/train', norm=1)
    data = ds_info.__getitem__(slice(0, ds_info.__len__(), 1))
