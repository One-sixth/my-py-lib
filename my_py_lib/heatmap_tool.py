'''
热图工具
'''

import numpy as np
from skimage.draw import disk as sk_disk


def get_cls_with_det_pts_from_cls_hm(cls_hm: np.ndarray, det_pts: np.ndarray, radius: int=3, mode: str='mean'):
    '''
    根据检测点和分类热图，从中获得获得每个检测点的类别
    :param cls_hm:  分类热图
    :param det_pts: 检测点坐标，要求形状为 [-1, 2]，坐标格式为 yx
    :param radius:  检索半径
    :param mode:    计算模式，转换检索半径内类别像素的到分类概率的方式
    :return:
    '''
    print('Warning! 该函数尚未使用过，需要测试')
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1
    assert mode in ('sum', 'mean', 'max', 'min')
    det_pts = np.asarray(det_pts, np.int32)
    if len(det_pts) == 0:
        det_pts = det_pts.reshape([-1, 2])
    assert det_pts.ndim == 2 and det_pts.shape[1] == 2
    radius = int(radius)
    # 每个点对应各个类别的概率
    cls_probs = np.zeros([len(det_pts), cls_hm.shape[2]], np.float32)
    hw = cls_hm.shape[:2]
    for i, pt in enumerate(det_pts):
        rr, cc = sk_disk(pt, radius=radius, shape=hw)
        for c in range(cls_hm.shape[2]):
            if mode == 'sum':
                cls_probs[i, c] = cls_hm[rr, cc, c].sum()
            elif mode == 'mean':
                cls_probs[i, c] = cls_hm[rr, cc, c].mean()
            elif mode == 'max':
                cls_probs[i, c] = cls_hm[rr, cc, c].max()
            elif mode == 'min':
                cls_probs[i, c] = cls_hm[rr, cc, c].min()
            else:
                raise AssertionError('Error! Invalid mode param.')
    return cls_probs