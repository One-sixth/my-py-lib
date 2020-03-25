'''
包围框工具箱，注意，输入和输出的包围框格式均为 y1x1y2x2
如果需要包围框格式转换，请使用 coord_tool 转换坐标格式
'''


import numpy as np


def resize_bbox(bbox: np.ndarray, factor_hw, center_yx=(0, 0)):
    '''
    基于指定位置对包围框进行缩放
    :param bbox:        要求 bbox 为 np.ndarray 和 格式为 y1x1y2x2
    :param factor_hw:   缩放倍率
    :param center_yx:   默认以(0, 0)为原点进行缩放
    :return:
    '''
    assert len(center_yx) == 2
    center_yxyx = np.array([*center_yx] * 2)
    bf = np.all(center_yxyx == (0, 0, 0, 0))
    if not bf:
        bbox = bbox - center_yxyx
        bbox = bbox * factor_hw
        bbox = bbox + center_yxyx
    else:
        bbox = bbox * factor_hw
    return bbox


def calc_bbox_iou_NtoN(bboxes1: np.ndarray, bboxes2: np.ndarray):
    '''
    计算N对个包围框的IOU，注意这里一一对应的，同时下面使用了...技巧，使其同时支持 1对N，N对1 的IOU计算
    注意不支持 N对M 的计算
    :param bboxes1:
    :param bboxes2:
    :return:
    '''
    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    inter_y1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    inter_x1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    inter_y2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    inter_x2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # 计算相交区域长宽
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_area = inter_h * inter_w

    uniou_area = bboxes1_area + bboxes2_area - inter_area
    # 如果分母为0时，使其等于1e-8，可以同时避免eps对iou干扰和除零问题
    uniou_area = np.where(uniou_area > 0, uniou_area, 1e-8)

    ious = inter_area / uniou_area
    return ious


def calc_bbox_iou_1toN(bbox1: np.ndarray, bboxes: np.ndarray):
    '''
    计算 1对N 的 IOU
    :param bbox1:
    :param bboxes:
    :return:
    '''
    assert bboxes.ndim == 2 and bbox1.ndim == 1
    return calc_bbox_iou_NtoN(bbox1, bboxes)
