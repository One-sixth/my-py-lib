'''
opencv 的单个轮廓格式为 [N, 1, xy]
我的单个轮廓格式为 [N, yx]

下面所有函数，除了轮廓格式转换函数，都是用我的轮廓格式输入和输出

注意，下面函数需要的坐标要求为整数，不支持浮点数运算
因为下面一些计算需要画图，浮点数无法画图。

'''

import cv2
import numpy as np
from typing import Iterable


def check_and_tr_umat(mat):
    '''
    如果输入了cv.UMAT，将会自动转换为 np.ndarray
    :param mat:
    :return:
    '''
    if isinstance(mat, cv2.UMat):
        mat = mat.get()
    return mat


def tr_cv_to_my_contours(cv_contours):
    '''
    轮廓格式转换，转换opencv格式到我的格式
    :param cv_contours:
    :return:
    '''
    out_contours = [c[:, 0, ::-1] for c in cv_contours]
    return out_contours


def tr_my_to_cv_contours(my_contours):
    '''
    轮廓格式转换，转换我的格式到opencv的格式
    :param my_contours:
    :return:
    '''
    out_contours = [c[:, None, ::-1] for c in my_contours]
    return out_contours


def calc_bbox_with_contour(contour):
    '''
    求轮廓的外接包围框
    :param contour:
    :return:
    '''
    min_y = np.min(contour[:, 0])
    max_y = np.max(contour[:, 0])
    min_x = np.min(contour[:, 1])
    max_x = np.max(contour[:, 1])
    bbox = np.array([min_y, min_x, max_y, max_x])
    return bbox


def simple_contours(contours, epsilon=0):
    '''
    简化轮廓，当俩个点的距离小于等于 epsilon 时，会融合这俩个点。
    epsilon=0 代表只消除重叠点
    :param contours:
    :param epsilon:
    :return:
    '''
    # 简化轮廓，目前用于缩放后去除重叠点，并不进一步简化
    # epsilon=0 代表只去除重叠点
    contours = tr_my_to_cv_contours(contours)
    out = []
    for c in contours:
        out.append(cv2.approxPolyDP(c, epsilon, True))
    out = tr_cv_to_my_contours(out)
    return out


def resize_contours(contours, scale_hw_factor=1.0):
    '''
    缩放轮廓
    :param contours: 输入一组轮廓
    :param scale_hw_factor: 缩放倍数
    :return:
    '''
    if isinstance(scale_hw_factor, Iterable):
        scale_hw_factor = np.array(scale_hw_factor)[None,]
    # 以左上角为原点进行缩放轮廓
    out = [(c * scale_hw_factor).astype(contours[0].dtype) for c in contours]
    return out


def calc_contour_area(contour):
    '''
    求轮廓的面积
    :param contour: 输入一个轮廓
    :return:
    '''
    c = tr_my_to_cv_contours([contour])[0]
    return cv2.contourArea(c)


def calc_convex_contours(coutours):
    '''
    计算一组轮廓的凸壳轮廓，很多时候可以加速。
    :param coutours:
    :return:
    '''
    coutours = tr_my_to_cv_contours(coutours)
    new_coutours = [cv2.convexHull(con) for con in coutours]
    new_coutours = tr_cv_to_my_contours(new_coutours)
    return new_coutours


def calc_iou_with_two_contours(contour1, contour2, max_test_area_hw=(512, 512)):
    '''
    求两个轮廓的IOU分数，使用绘图法，速度相当慢。
    :param contour1: 轮廓1
    :param contour2: 轮廓2
    :param max_test_area_hw: 最大缓存区大小，若计算的轮廓大小大于限制，则会先缩放轮廓
    :return:
    '''
    # 计算俩个轮廓的iou
    bbox1 = calc_bbox_with_contour(contour1)
    bbox2 = calc_bbox_with_contour(contour2)
    merge_bbox_y1x1 = np.where(bbox1 < bbox2, bbox1, bbox2)[:2]
    merge_bbox_y2x2 = np.where(bbox1 > bbox2, bbox1, bbox2)[2:]
    contour1 = contour1 - merge_bbox_y1x1
    contour2 = contour2 - merge_bbox_y1x1
    merge_bbox_y2x2 -= merge_bbox_y1x1
    merge_bbox_y1x1.fill(0)
    merge_bbox_hw = merge_bbox_y2x2
    if max_test_area_hw is not None:
        hw_factor = merge_bbox_hw / max_test_area_hw
        max_factor = np.max(hw_factor)
        if max_factor > 1:
            contour1, contour2 = resize_contours([contour1, contour2], 1 / max_factor)
            merge_bbox_hw = np.ceil(merge_bbox_hw / max_factor).astype(np.int32)
    reg = np.zeros(merge_bbox_hw.astype(np.int), np.uint8)
    reg1 = draw_contours(reg.copy(), [contour1], 1, -1)
    reg2 = draw_contours(reg.copy(), [contour2], 1, -1)
    cross_reg = np.all([reg1, reg2], 0).astype(np.uint8)
    cross_contours, _ = cv2.findContours(cross_reg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cross_area = 0
    for c in cross_contours:
        cross_area += cv2.contourArea(c)
    contour1_area = calc_contour_area(contour1)
    contour2_area = calc_contour_area(contour2)
    iou = cross_area / (contour1_area + contour2_area - cross_area + 1e-8)
    return iou


def draw_contours(im, contours, color, thickness=2, auto_cvt_convex=False):
    '''
    绘制轮廓
    :param im:  图像
    :param contours: 多个轮廓的列表，格式为 [(n,pt_yx), (n,pt_yx)]
    :param color: 绘制的颜色
    :param thickness: 绘制轮廓边界大小
    :param auto_cvt_convex: 是否将轮廓转换为凸壳后再绘制
    :return:
    '''
    if isinstance(color, Iterable):
        dim2 = len(color)
        color = tuple(color)
    else:
        dim2 = 1
        color = (color,)
    if auto_cvt_convex:
        contours = calc_convex_contours(contours)
    contours = tr_my_to_cv_contours(contours)
    im = cv2.drawContours(im, contours, -1, color, thickness)
    im = check_and_tr_umat(im)
    return im


def calc_one_contour_with_multi_contours_iou(c1, batch_c):
    '''
    求一个轮廓和一组轮廓的IOU分数
    :param c1:
    :param batch_c:
    :return:
    '''
    ious = np.zeros([len(batch_c)], np.float32)
    for i, c2 in enumerate(batch_c):
        iou = calc_iou_with_two_contours(c1, c2)
        ious[i] = iou
    return ious


def fusion_im_contours(im, contours, classes, class_to_color_map):
    '''
    融合原图和一组轮廓到一张图像
    :param im:                  输入图像，要求为 np.array
    :param contours:            输入轮廓
    :param classes:             每个轮廓的类别
    :param class_to_color_map:  每个类别的颜色
    :return:
    '''
    assert set(classes).issubset(set(class_to_color_map.keys()))
    im = im.copy()
    contours = np.array(contours)
    clss = np.asarray(classes)
    for cls in set(clss):
        cs = contours[clss == cls]
        im = draw_contours(im, list(cs), class_to_color_map[cls])
    return im


def merge_contours(contours, use_convex=True):
    '''
    合并俩个轮廓，当前只支持凸壳轮廓合并
    :param contours: 多个轮廓，要求格式为我的格式
    :param use_convex: 是否使用凸壳合并
    :return:
    '''
    assert use_convex, 'Now only suport convex set to True.'
    points = np.concatenate(contours, 0)
    points = calc_convex_contours([points])[0]
    return points


if __name__ == '__main__':
    c1 = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], np.int)
    c2 = c1.copy()
    c2[:, 1] += 50
    print(calc_contour_area(c1))
    print(calc_iou_with_two_contours(c1, c2))
    print(calc_one_contour_with_multi_contours_iou(c1, [c1,c2]))
    print(merge_contours([c1, c2]))
    im = np.zeros([512, 512, 3], np.uint8)
    im2 = draw_contours(im, [c1, c2], [0, 0, 255], 2)
    im3 = fusion_im_contours(im, [c1, c2], [0, 1], {0: [255, 0, 0], 1: [0, 255, 0]})
    cv2.imshow('show', im2[..., ::-1])
    cv2.imshow('show2', im3[..., ::-1])
    cv2.waitKey(0)
