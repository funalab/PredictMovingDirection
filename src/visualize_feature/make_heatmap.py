import cv2
import numpy as np

colormaps = {
    'jet': cv2.COLORMAP_JET,
    'summer': cv2.COLORMAP_SUMMER
}


def cvt_rel(rel, c='summer'):
    rel_map = rel.copy()
    if len(rel_map.shape) == 3:
        rel_map = rel_map.reshape(rel_map.shape[2], rel_map.shape[3])
    max_rel = rel_map.max()
    min_rel = rel_map.min()
    rel_map = (rel_map - min_rel) / (max_rel - min_rel) * 255
    rel_map = rel_map.astype(np.uint8)
    rel_map = cv2.applyColorMap(rel_map, colormaps[c])

    return rel_map
