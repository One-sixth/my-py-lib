'''
包围框评分工具
'''

import numpy as np
from .score_tool import calc_score_f05_f1_f2_prec_recall
from .bbox_tool import calc_bbox_iou_1toN


def calc_bbox_score(pred_bboxes, pred_cls, label_bboxes, label_cls, cls_list, match_iou_threshs=(0.3, 0.5, 0.7), use_single_pair=False):
    '''
    通用关键点评估
    将会返回一个分数字典
    当预测与标签距离小于评估距离时，将会认定为真阳性
    结构为
    类别-评估距离-X
    X:
    found_pred      真阳性，预测正确的数量
    fakefound_pred  假阳性，预测失败的数量
    found_label     真阳性，标签被正确匹配到的数量
    nofound_label   假阴性，没有任何成功匹配的标签数量
    pred_repeat     当use_single_pair为False时，一个预测可以同时匹配多个标签，该度量将会统计匹配数量大于1的预测的数量
    label_repeat    当use_single_pair为False时，一个标签可以同时匹配多个预测，该度量将会统计匹配数量大于1的标签的数量

    :param pred_bboxes:         预测的包围框
    :param pred_cls:            预测的类别
    :param label_bboxes:        标签的包围框
    :param label_cls:           标签的类别
    :param cls_list:            要评估的类别列表
    :param match_iou_threshs:   多个评估阈值
    :param use_single_pair:     若为真，则使用一个预测只匹配一个标签。如果假，每个预测都可以匹配多个标签
    :return:
    '''
    score_table = {}

    if len(pred_bboxes) == 0 or len(label_bboxes) == 0:
        for cls in cls_list:
            score_table[cls] = {}
            for iou_th in match_iou_threshs:
                score_table[cls][iou_th] = {}
                score_table[cls][iou_th]['found_pred'] = 0
                score_table[cls][iou_th]['fakefound_pred'] = len(pred_bboxes)
                score_table[cls][iou_th]['found_label'] = 0
                score_table[cls][iou_th]['nofound_label'] = len(label_bboxes)
                score_table[cls][iou_th]['pred_repeat'] = 0
                score_table[cls][iou_th]['label_repeat'] = 0
                score_table[cls][iou_th]['f05'] = 0.
                score_table[cls][iou_th]['f1'] = 0.
                score_table[cls][iou_th]['f2'] = 0.
                score_table[cls][iou_th]['prec'] = 0.
                score_table[cls][iou_th]['recall'] = 0.
        return score_table

    pred_bboxes = np.asarray(pred_bboxes, np.float32)
    label_bboxes = np.asarray(label_bboxes, np.float32)

    pred_cls = np.asarray(pred_cls, np.int32)
    label_cls = np.asarray(label_cls, np.int32)

    assert pred_bboxes.ndim == 2 and pred_bboxes.shape[1] == 4
    assert label_bboxes.ndim == 2 and label_bboxes.shape[1] == 4
    assert pred_cls.ndim == 1
    assert label_cls.ndim == 1
    assert len(pred_bboxes) == len(pred_cls)
    assert len(label_bboxes) == len(label_cls)

    for cls in cls_list:
        score_table[cls] = {}
        pred_selected_bools = np.array(pred_cls, np.int32) == cls
        label_selected_bools = np.array(label_cls, np.int32) == cls
        selected_pred_bboxes = pred_bboxes[pred_selected_bools]
        selected_label_bboxes = label_bboxes[label_selected_bools]

        for iou_th in match_iou_threshs:
            score_table[cls][iou_th] = {}

            label_found_count = np.zeros(len(selected_label_bboxes), np.int32)
            pred_found_count = np.zeros(len(selected_pred_bboxes), np.int32)

            if not use_single_pair:
                for pi, pred_bbox in enumerate(selected_pred_bboxes):
                    if len(selected_label_bboxes) != 0:
                        ious = calc_bbox_iou_1toN(pred_bbox, selected_label_bboxes)
                        close_bools = ious >= iou_th
                        label_found_count[close_bools] += 1
                        pred_found_count[pi] += np.array(close_bools, np.int32).sum()
            else:
                for pi, pred_bbox in enumerate(selected_pred_bboxes):
                    if len(selected_label_bboxes) != 0:
                        ious = calc_bbox_iou_1toN(pred_bbox, selected_label_bboxes)
                        close_bools = ious >= iou_th
                        for ii in np.argsort(ious)[::-1]:
                            if not close_bools[ii]:
                                break
                            if label_found_count[ii] > 0:
                                continue
                            else:
                                label_found_count[ii] += 1
                                pred_found_count[pi] += 1

            found_pred = (pred_found_count > 0).sum()
            fakefound_pred = (pred_found_count == 0).sum()

            found_label = (label_found_count > 0).sum()
            nofound_label = (label_found_count == 0).sum()

            pred_repeat = (pred_found_count > 1).sum()
            label_repeat = (label_found_count > 1).sum()

            f05, f1, f2, prec, recall = calc_score_f05_f1_f2_prec_recall(found_label, nofound_label, found_pred, fakefound_pred)

            score_table[cls][iou_th]['found_pred'] = int(found_pred)
            score_table[cls][iou_th]['fakefound_pred'] = int(fakefound_pred)
            score_table[cls][iou_th]['found_label'] = int(found_label)
            score_table[cls][iou_th]['nofound_label'] = int(nofound_label)
            score_table[cls][iou_th]['pred_repeat'] = int(pred_repeat)
            score_table[cls][iou_th]['label_repeat'] = int(label_repeat)
            score_table[cls][iou_th]['f05'] = float(f05)
            score_table[cls][iou_th]['f1'] = float(f1)
            score_table[cls][iou_th]['f2'] = float(f2)
            score_table[cls][iou_th]['prec'] = float(prec)
            score_table[cls][iou_th]['recall'] = float(recall)

    return score_table


def summary_bbox_score(scores, cls_list, match_iou_thresh_list):
    '''
    对多个分数表进行合算，得到统计分数表
    其中 found_pred, fakefound_pred, found_label, nofound_label, pred_repeat, label_repeat 将会被累加
    其中 f1, prec, recall 将会被求平均
    :param scores:                      多个分数表
    :param cls_list:                    要检查的分类
    :param match_iou_thresh_list:       多个匹配IOU
    :return:
    '''
    score_table = {}
    for cls in cls_list:
        score_table[cls] = {}
        for iou_th in match_iou_thresh_list:
            score_table[cls][iou_th] = {}
            score_table[cls][iou_th]['found_pred'] = 0
            score_table[cls][iou_th]['fakefound_pred'] = 0
            score_table[cls][iou_th]['found_label'] = 0
            score_table[cls][iou_th]['nofound_label'] = 0
            score_table[cls][iou_th]['pred_repeat'] = 0
            score_table[cls][iou_th]['label_repeat'] = 0
            score_table[cls][iou_th]['f05'] = 0.
            score_table[cls][iou_th]['f1'] = 0.
            score_table[cls][iou_th]['f2'] = 0.
            score_table[cls][iou_th]['prec'] = 0.
            score_table[cls][iou_th]['recall'] = 0.

    for score in scores:
        for cls in cls_list:
            for iou_th in match_iou_thresh_list:
                score_table[cls][iou_th]['found_pred'] += score[cls][iou_th]['found_pred']
                score_table[cls][iou_th]['fakefound_pred'] += score[cls][iou_th]['fakefound_pred']
                score_table[cls][iou_th]['found_label'] += score[cls][iou_th]['found_label']
                score_table[cls][iou_th]['nofound_label'] += score[cls][iou_th]['nofound_label']
                score_table[cls][iou_th]['pred_repeat'] += score[cls][iou_th]['pred_repeat']
                score_table[cls][iou_th]['label_repeat'] += score[cls][iou_th]['label_repeat']
                score_table[cls][iou_th]['f05'] += score[cls][iou_th]['f05'] / len(scores)
                score_table[cls][iou_th]['f1'] += score[cls][iou_th]['f1'] / len(scores)
                score_table[cls][iou_th]['f2'] += score[cls][iou_th]['f2'] / len(scores)
                score_table[cls][iou_th]['prec'] += score[cls][iou_th]['prec'] / len(scores)
                score_table[cls][iou_th]['recall'] += score[cls][iou_th]['recall'] / len(scores)

    return score_table
