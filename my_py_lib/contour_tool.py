'''
opencv 的单个轮廓格式为 [N, 1, xy]
我的单个轮廓格式为 [N, yx]

下面所有函数，除了轮廓格式转换函数和部分函数，都是用我的轮廓格式输入和输出
函数名以shapely开头的，需要输入polygon格式

注意，下面函数需要的坐标要求为整数，不支持浮点数运算
因为下面一些计算需要画图，浮点数无法画图。

现在，可以支持浮点数了，现在使用shapely库用作轮廓运算

优先使用opencv的函数处理，opencv没有的才使用shapely，因为opencv函数的速度比shapely快

'''

import cv2
assert cv2.__version__ >= '4.0'
import numpy as np
from typing import Iterable, List
from shapely.geometry import Polygon
from shapely.ops import unary_union
try:
    from im_tool import ensure_image_has_same_ndim
except ModuleNotFoundError:
    from .im_tool import ensure_image_has_same_ndim
# from list_tool import list_multi_get_with_ids, list_multi_get_with_bool


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


def tr_my_to_polygon(my_contours):
    '''
    轮廓格式转换，转换我的格式到polygon
    :param my_contours:
    :return:
    '''
    polygons = []
    for c in my_contours:
        p = Polygon(c[:, ::-1])
        p = p if p.is_valid else p.buffer(0)
        polygons.append(p)
    return polygons


def tr_polygons_to_my(polygons: List[Polygon], dtype=np.float32):
    '''
    转换shapely的多边形到我的格式
    :param polygons:
    :param dtype: 输出数据类型
    :return:
    '''
    my_contours = []
    for poly in polygons:
        x, y = poly.exterior.xy
        c = np.array(list(zip(y, x)), dtype)
        my_contours.append(c)
    return my_contours


def tr_polygons_to_cv(polygons: List[Polygon], dtype=np.float32):
    '''
    转换shapely的多边形到opencv格式
    :param polygons:
    :param dtype:   输出数据类型
    :return:
    '''
    cv_contours = []
    for poly in polygons:
        x, y = poly.exterior.xy
        c = np.array(list(zip(x, y)), dtype)[:, None]
        cv_contours.append(c)
    return cv_contours


def find_contours(im, mode, method):
    '''
    cv2.findContours 的包装，区别是会自动转换cv2的轮廓格式到我的格式，和会自动删除轮廓点少于3的无效轮廓
    :param im:
    :param mode: 轮廓查找模式，例如 cv2.RETR_EXTERNAL cv2.RETR_TREE, cv2.RETR_LIST
    :param method: 轮廓优化方法，例如 cv2.CHAIN_APPROX_SIMPLE cv2.CHAIN_APPROX_NONE
    :return:
    '''
    # 简化轮廓，目前用于缩放后去除重叠点，并不进一步简化
    # epsilon=0 代表只去除重叠点
    contours, _ = cv2.findContours(im, mode=mode, method=method)
    # 删除轮廓点少于3的无效轮廓
    valid_contours = []
    for c in contours:
        if len(c) >= 3:
            valid_contours.append(c)
    valid_contours = tr_cv_to_my_contours(valid_contours)
    return valid_contours


def offset_contours(contours, ori_yx=(0, 0), new_ori_yx=(0, 0)):
    '''
    重定位轮廓位置
    :param contours:    多个轮廓，要求为我的格式
    :param ori_yx:      旧轮廓的起点
    :param new_ori_yx:  新起点
    :return:
    '''
    if len(contours) == 0:
        return contours
    new_contours = []
    ori_yx = np.array(ori_yx, dtype=contours[0].dtype).reshape(1, 2)
    new_ori_yx = np.array(new_ori_yx, dtype=contours[0].dtype).reshape(1, 2)
    for c in contours:
        c = c - ori_yx + new_ori_yx
        new_contours.append(c)
    return new_contours


def make_bbox_to_contour(start_yx=(0, 0), bbox_hw=(1, 1)):
    '''
    将包围框转换为轮廓
    :param bbox: np.ndarray []
    :return:
    '''
    p1 = [start_yx[0]               , start_yx[1]             ]
    p2 = [start_yx[0]               , start_yx[1] + bbox_hw[1]]
    p3 = [start_yx[0] + bbox_hw[0]  , start_yx[1] + bbox_hw[1]]
    p4 = [start_yx[0] + bbox_hw[0]  , start_yx[1]             ]
    contours = [p1, p2, p3, p4]
    contours = np.array(contours)
    return contours


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


def resize_contours(contours, scale_factor_hw=1.0):
    '''
    缩放轮廓
    :param contours: 输入一组轮廓
    :param scale_factor_hw: 缩放倍数
    :return:
    '''
    if isinstance(scale_factor_hw, Iterable):
        scale_factor_hw = np.array(scale_factor_hw)[None,]
    # 以左上角为原点进行缩放轮廓
    out = [(c * scale_factor_hw).astype(contours[0].dtype) for c in contours]
    return out


def calc_contour_area(contour: np.ndarray):
    '''
    求轮廓的面积
    :param contour: 输入一个轮廓
    :return:
    '''
    area = cv2.contourArea(tr_my_to_cv_contours([contour])[0])
    return area


def calc_convex_contours(coutours):
    '''
    计算一组轮廓的凸壳轮廓，很多时候可以加速。
    :param coutours: 一组轮廓
    :return: 返回一组自身的凸壳轮廓
    '''
    coutours = tr_my_to_cv_contours(coutours)
    new_coutours = [cv2.convexHull(con) for con in coutours]
    new_coutours = tr_cv_to_my_contours(new_coutours)
    return new_coutours


def shapely_calc_distance_contour(polygon1: Polygon, polygon2: Polygon):
    '''
    求俩个轮廓的最小距离，原型
    :param polygon1: 多边形1
    :param polygon2: 多边形2
    :return: 轮廓面积
    '''
    l = polygon1.distance(polygon2)
    return l


# def calc_iou_with_two_contours(contour1, contour2, max_test_area_hw=(512, 512)):
#     '''
#     求两个轮廓的IOU分数，使用绘图法，速度相当慢。
#     :param contour1: 轮廓1
#     :param contour2: 轮廓2
#     :param max_test_area_hw: 最大缓存区大小，若计算的轮廓大小大于限制，则会先缩放轮廓
#     :return:
#     '''
#     # 计算俩个轮廓的iou
#     bbox1 = calc_bbox_with_contour(contour1)
#     bbox2 = calc_bbox_with_contour(contour2)
#     merge_bbox_y1x1 = np.where(bbox1 < bbox2, bbox1, bbox2)[:2]
#     merge_bbox_y2x2 = np.where(bbox1 > bbox2, bbox1, bbox2)[2:]
#     contour1 = contour1 - merge_bbox_y1x1
#     contour2 = contour2 - merge_bbox_y1x1
#     merge_bbox_y2x2 -= merge_bbox_y1x1
#     merge_bbox_y1x1.fill(0)
#     merge_bbox_hw = merge_bbox_y2x2
#     if max_test_area_hw is not None:
#         hw_factor = merge_bbox_hw / max_test_area_hw
#         max_factor = np.max(hw_factor)
#         if max_factor > 1:
#             contour1, contour2 = resize_contours([contour1, contour2], 1 / max_factor)
#             merge_bbox_hw = np.ceil(merge_bbox_hw / max_factor).astype(np.int32)
#     reg = np.zeros(merge_bbox_hw.astype(np.int), np.uint8)
#     reg1 = draw_contours(reg.copy(), [contour1], 1, -1)
#     reg2 = draw_contours(reg.copy(), [contour2], 1, -1)
#     cross_reg = np.all([reg1, reg2], 0).astype(np.uint8)
#     cross_contours, _ = cv2.findContours(cross_reg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cross_area = 0
#     for c in cross_contours:
#         cross_area += cv2.contourArea(c)
#     contour1_area = calc_contour_area(contour1)
#     contour2_area = calc_contour_area(contour2)
#     iou = cross_area / (contour1_area + contour2_area - cross_area + 1e-8)
#     return iou


def shapely_calc_iou_with_two_contours(contour1, contour2):
    '''
    计算俩个轮廓的IOU，原型
    :param contour1: polygon多边形1
    :param contour2: polygon多边形1
    :return:    IOU分数
    '''
    c1 = contour1
    c2 = contour2
    if not c1.intersects(c2):
        return 0.
    area1 = c1.area
    area2 = c2.area
    inter_area = c1.intersection(c2).area
    iou = inter_area / (area1 + area2 - inter_area + 1e-8)
    return iou


def calc_iou_with_two_contours(contour1, contour2):
    '''
    计算两个轮廓的IOU
    :param contour1: 轮廓1
    :param contour2: 轮廓2
    :return:
    '''
    c1, c2 = tr_my_to_polygon([contour1, contour2])
    iou = shapely_calc_iou_with_two_contours(c1, c2)
    return iou


def draw_contours(im, contours, color, thickness=2):
    '''
    绘制轮廓
    :param im:  图像
    :param contours: 多个轮廓的列表，格式为 [(n,pt_yx), (n,pt_yx)]
    :param color: 绘制的颜色
    :param thickness: 绘制轮廓边界大小
    :return:
    '''
    ori_im = im
    if isinstance(color, Iterable):
        color = tuple(color)
    else:
        color = (color,)
        if im.ndim == 3:
            color = color * im.shape[-1]
    contours = tr_my_to_cv_contours(contours)
    # 确保为整数
    contours = [c.astype(np.int32) for c in contours]
    im = cv2.drawContours(im, contours, -1, color, thickness)
    im = check_and_tr_umat(im)
    im = ensure_image_has_same_ndim(im, ori_im)
    return im


def shapely_calc_distance_with_contours_1toN(c1: Polygon, batch_c: Iterable[Polygon]):
    '''
    求一个轮廓和一组轮廓的IOU分数，原型
    :param c1:
    :param batch_c:
    :return:
    '''
    ious = np.zeros([len(batch_c)], np.float32)
    for i, c2 in enumerate(batch_c):
        ious[i] = c1.distance(c2)
    return ious


def shapely_calc_iou_with_contours_1toN(c1: Polygon, batch_c: Iterable[Polygon]):
    '''
    求一个轮廓和一组轮廓的IOU分数，原型
    :param c1:
    :param batch_c:
    :return:
    '''
    ious = np.zeros([len(batch_c)], np.float32)
    for i, c2 in enumerate(batch_c):
        ious[i] = shapely_calc_iou_with_two_contours(c1, c2)
    return ious


def calc_iou_with_contours_1toN(c1, batch_c):
    '''
    求一个轮廓和一组轮廓的IOU分数，包装
    :param c1:
    :param batch_c:
    :return:
    '''
    c1 = tr_my_to_polygon([c1])[0]
    batch_c = tr_my_to_polygon(batch_c)
    ious = shapely_calc_iou_with_contours_1toN(c1, batch_c)
    return ious


def fusion_im_contours(im, contours, classes, class_to_color_map, copy=True):
    '''
    融合原图和一组轮廓到一张图像
    :param im:                  输入图像，要求为 np.array
    :param contours:            输入轮廓
    :param classes:             每个轮廓的类别
    :param class_to_color_map:  每个类别的颜色
    :return:
    '''
    assert set(classes).issubset(set(class_to_color_map.keys()))
    if copy:
        im = im.copy()
    contours = np.array(contours)
    clss = np.asarray(classes)
    for cls in set(clss):
        cs = contours[clss == cls]
        im = draw_contours(im, list(cs), class_to_color_map[cls])
    return im


def shapely_merge_to_single_contours(polygons: List[Polygon]):
    '''
    合并多个轮廓到一个轮廓，以第一个轮廓为主，不以第一个轮廓相交的其他轮廓将会被忽略，原型
    :param polygons: 多个多边形，其中第一个为需要融合的主体，后面的如果与第一个不相交，则会忽略
    :return: 已合并的轮廓
    '''
    ps = list(polygons)
    # 逆序删除，避免问题
    # range不能逆序，需要先转换为list
    for i in list(range(1, len(ps)))[::-1]:
        if not ps[0].intersects(ps[i]):
            del ps[i]
    p = unary_union(ps)
    return p


def merge_to_single_contours(contours, auto_simple=True):
    '''
    合并多个轮廓到一个轮廓，以第一个轮廓为主，不以第一个轮廓相交的其他轮廓将会被忽略
    :param contours: 多个轮廓
    :param auto_simple: 是否自动优化生成的轮廓
    :return: 一个轮廓
    '''
    ps = tr_my_to_polygon(contours)
    p = shapely_merge_to_single_contours(ps)
    c = tr_polygons_to_my([p], contours[0].dtype)[0]
    if auto_simple:
        c = simple_contours([c], epsilon=0)[0]
    return c


# def merge_multi_contours_sort_by_area(contours1):
#     '''
#     轮廓后处理，方法：按轮廓面积排序，以感染的方式融合其他相交的轮廓。
#     简而言之，融合相交的轮廓，分类取面积最大的轮廓类别。
#     :param contours:    多个轮廓，要求为我的格式
#     :return:
#     '''
#     polygons = tr_my_to_polygon(contours1)
#
#     cons_area = [p.area for p in polygons]
#     sorted_ids = np.argsort(cons_area)[::-1]
#
#     open_ids = list(sorted_ids)             # 尚未处理的轮廓ID
#     close_ids = []                          # 独立的轮廓ID
#     remove_ids = []                         # 被移除的轮廓ID
#
#     while len(open_ids) > 0:
#         cid = open_ids.pop(0)
#         close_ids.append(cid)
#
#         poly = polygons[cid]
#         l1 = shapely_calc_one_contour_with_multi_contours_distance(poly, list_multi_get_with_ids(polygons, open_ids))
#         wait_to_merge_ids = list_multi_get_with_bool(open_ids, l1 > 0)      # 等待融合的ID
#         if len(wait_to_merge_ids) > 0:
#             for i in wait_to_merge_ids:
#                 open_ids.remove(i)
#                 remove_ids.append(i)
#             del close_ids[-1]
#             open_ids.insert(0, cid)
#             wait_to_merge_con = list_multi_get_with_ids(polygons, wait_to_merge_ids)
#             wait_to_merge_con.insert(0, poly)
#             polygons[cid] = shapely_merge_to_single_contours(wait_to_merge_con)
#
#     polygons = list_multi_get_with_ids(polygons, close_ids)
#     contours = tr_polygons_to_my(polygons, np.float32)
#
#     return contours, close_ids


def shapely_merge_multi_contours(polygons: List[Polygon]):
    '''
    直接合并多个轮廓
    :param polygons: 多个轮廓，要求为shapely的格式
    :return:
    '''
    mps = unary_union(polygons)
    if isinstance(mps, Polygon):
        mps = [mps]
    return mps


def merge_multi_contours_sort_by_area(contours):
    '''
    轮廓后处理2，方法：直接合并所有相交的轮廓，计算合并后的轮廓与原始轮廓的IOU，取最大IOU的原始轮廓ID
    :param contours: 多个轮廓，要求为我的格式
    :return: 返回合并后的剩余轮廓，和保留轮廓位于原列表时的ID
    '''
    polygons = tr_my_to_polygon(contours)
    mps = shapely_merge_multi_contours(polygons)
    ids = []
    for mp in mps:
        ious = shapely_calc_iou_with_contours_1toN(mp, polygons)
        ids.append(np.argmax(ious))

    cs = tr_polygons_to_my(mps)
    return cs, ids


def merge_multi_contours(contours):
    '''
    直接合并多个轮廓
    :param contours: 多个轮廓，要求为我的格式
    :return: 返回合并后的剩余轮廓
    '''
    polygons = tr_my_to_polygon(contours)
    mps = shapely_merge_multi_contours(polygons)
    cs = tr_polygons_to_my(mps)
    return cs, ids


if __name__ == '__main__':
    c1 = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], np.int)
    c2 = c1.copy()
    c2[:, 1] += 50
    print(calc_contour_area(c1))
    print(calc_iou_with_two_contours(c1, c2))
    print(calc_iou_with_contours_1toN(c1, [c1,c2]))
    print(merge_to_single_contours([c1, c2]))
    im = np.zeros([512, 512, 3], np.uint8)
    im2 = draw_contours(im, [c1, c2], [0, 0, 255], 2)
    im3 = fusion_im_contours(im, [c1, c2], [0, 1], {0: [255, 0, 0], 1: [0, 255, 0]})
    cv2.imshow('show', im2[..., ::-1])
    cv2.imshow('show2', im3[..., ::-1])
    cv2.waitKey(0)
