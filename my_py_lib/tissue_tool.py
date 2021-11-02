'''
从病理图中获得主要组织区域的轮廓。
'''

import cv2
import numpy as np
try:
    from .contour_tool import find_contours, draw_contours, resize_contours
except ModuleNotFoundError:
    from contour_tool import find_contours, draw_contours, resize_contours


def get_tissue_contours(im, gray_thresh=210, area_thresh=0.005, *, debug_show=False):
    """
    从一张图像中获得组织轮廓
    :param im:                  opsl缩略图，要求为RGB的nd.ndarray类型
    :param gray_thresh:         灰度阈值
    :param area_thresh:         区域占比阈值，小于该比例的轮廓将会被丢弃
    :param debug_show:          是否显示调试数据
    :return: tissue_contours
    """

    tim1 = np.any(im <= gray_thresh, 2).astype(np.uint8)
    tim2 = (np.max(im, 2) - np.min(im, 2) > 18).astype(np.uint8)
    tim = tim1 * tim2

    # gim = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # tim = (gim < gray_thresh).astype(np.uint8)

    if debug_show:
        cv2.imshow('get_tissue_ori', im)
        # cv2.imshow('get_tissue_gim', gim)
        cv2.imshow('get_tissue_tim', tim*255)
        cv2.imshow('get_tissue_tim1', tim1*255)
        cv2.imshow('get_tissue_tim2', tim2*255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.dilate(tim, k, dst=tim, iterations=2)
    cv2.erode(tim, k, dst=tim, iterations=4)
    cv2.dilate(tim, k, dst=tim, iterations=2)

    if debug_show:
        cv2.imshow('get_tissue_tim3', tim*255)

    contours = find_contours(tim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tissue_contours = []

    for i, cont in enumerate(contours):
        contour_area = cv2.contourArea(cont)
        factor = float(contour_area) / np.prod(im.shape[:2], dtype=np.float)
        if debug_show:
            print(contour_area, '{:.3f}'.format(factor))
        if factor > area_thresh:
            tissue_contours.append(cont)

    if debug_show:
        mask = np.zeros([im.shape[0], im.shape[1], 1], dtype=np.uint8)
        mask = draw_contours(mask, tissue_contours, [1], thickness=-1)
        cv2.imshow('mask', mask*255)

    return tissue_contours


def get_tissue_contours_with_big_pic(opsl_im, gray_thresh=210, area_thresh=0.005, *, debug_show=False):
    '''
    对大图获得组织轮廓，默认参数即可对HE图像工作良好
    :param opsl_im:
    :param gray_thresh:
    :param area_thresh:
    :param debug_show:  设定为True可以启动调试功能
    :return:
    '''
    thumb = opsl_im.get_thumbnail([768, 768])
    thumb = np.array(thumb)
    tissue_contours = get_tissue_contours(thumb, gray_thresh=gray_thresh, area_thresh=area_thresh, debug_show=debug_show)
    thumb_hw = thumb.shape[:2]
    factor_hw = np.array(opsl_im.level_dimensions[0][::-1], dtype=np.float32) / thumb_hw
    resized_contours = resize_contours(tissue_contours, factor_hw)
    return resized_contours


if __name__ == '__main__':
    import glob
    import os
    if os.name == 'nt':
        openslide_bin = os.path.split(__file__)[0]+'/../bin_openslide_x64_20171122'
        os.putenv('PATH', os.getenv('PATH') + ';' + openslide_bin)
        os.add_dll_directory(openslide_bin)
    import openslide as opsl

    for im_path in glob.glob('dataset/ims/*.ndpi', recursive=True):
        im = opsl.OpenSlide(im_path)
        get_tissue_contours_with_big_pic(im, debug_show=True)
        cv2.waitKey(0)
