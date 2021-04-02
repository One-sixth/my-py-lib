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
        bbox[:2] *= factor_hw
        bbox[2:] *= factor_hw
        bbox = bbox + center_yxyx
    else:
        bbox[:2] *= factor_hw
        bbox[2:] *= factor_hw
    return bbox


def calc_bbox_center(bbox: np.ndarray):
    '''
    基于指定位置对包围框进行缩放
    :param bbox:        要求 bbox 为 np.ndarray 和 格式为 y1x1y2x2
    :param factor_hw:   缩放倍率
    :param center_yx:   默认以(0, 0)为原点进行缩放
    :return:
    '''
    assert isinstance(bbox, np.ndarray)
    assert len(bbox) == 4
    center = np.asarray((bbox[:2] + bbox[2:]) / 2, dtype=bbox.dtype)
    return center


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


def pad_bbox_to_square(bbox: np.ndarray):
    '''
    将输入的包围框填充为正方形，要求包围框坐标格式为y1x1y2x2或x1y1x2y2
    :param bbox:
    :return:
    '''
    assert bbox.ndim == 1 and len(bbox) == 4
    dtype = bbox.dtype
    bbox = np.asarray(bbox, np.float32)
    t1 = bbox[2:] - bbox[:2]
    tmax = max(t1)
    t2 = tmax / t1
    c = (bbox[:2] + bbox[2:]) / 2
    bbox = resize_bbox(bbox, t2, c)
    bbox = np.asarray(bbox, dtype)
    return bbox


def nms_process(confs: np.ndarray, bboxes: np.ndarray, iou_thresh: float=0.7):
    '''
    NMS 过滤，只需要置信度和坐标
    :param confs:       置信度列表，形状为 [-1] 或 [-1, 1]
    :param bboxes:      包围框列表，形状为 [-1, 4]，坐标格式为 y1x1y2x2 或 x1y1x2y2
    :param iou_thresh:  IOU阈值
    :return:
    '''
    assert len(confs) == len(bboxes)
    confs = np.asarray(confs, np.float32).reshape([-1])
    bboxes = np.asarray(bboxes, np.float32).reshape([-1, 4])
    assert len(confs) == len(bboxes)
    ids = np.argsort(confs)[::-1]
    ids = ids.reshape([-1])
    keep_boxes = []
    keep_ids = []
    for i in ids:
        if len(keep_boxes) == 0:
            keep_boxes.append(bboxes[i])
            keep_ids.append(i)
            continue
        cur_box = bboxes[i]
        ious = calc_bbox_iou_1toN(cur_box, np.asarray(keep_boxes, np.float32))
        max_iou = np.max(ious)
        if max_iou > iou_thresh:
            continue
        keep_boxes.append(bboxes[i])
        keep_ids.append(i)
    return keep_ids
