import numpy as np


def get_shortest_link_pair(pt_list1, pt_list2, dist_th: float):
    '''
    求两组点之间最短链接对
    :param pt_list1: 点集1
    :param pt_list2: 点集2
    :param dist_th:  最长配对距离，最短距离超过该阈值后不进行配对。
    :return: 返回点对的编号和他们的距离
    '''
    pt_list1 = np.asarray(pt_list1, np.float32).reshape(-1, 2)
    pt_list2 = np.asarray(pt_list2, np.float32).reshape(-1, 2)

    contact_pts1 = []
    contact_pts2 = []
    contact_distance = []

    if 0 == len(pt_list1) or 0 == len(pt_list2):
        return contact_pts1, contact_pts2, contact_distance

    # 先求出所有可以关联的链接，使用numpy的方法，去掉一个循环
    for pt1_id, pt1 in enumerate(pt_list1):
        ds = np.linalg.norm(pt1[None] - pt_list2, 2, axis=1)
        bs = ds <= dist_th
        pt2_ids = np.argwhere(bs).reshape(-1)

        contact_pts1.extend([pt1_id] * len(pt2_ids))
        contact_pts2.extend(pt2_ids)
        contact_distance.extend(ds[pt2_ids])

        # for pt2_id, pt2 in enumerate(pt_list2):
        #     pt2 = np.array(pt2, np.float32)
        #     d = np.linalg.norm(pt1 - pt2, 2)
        #     if d <= dist_th:
        #         contact_pts1.append(pt1_id)
        #         contact_pts2.append(pt2_id)
        #         contact_distance.append(d)

    out_contact_pts1 = []
    out_contact_pts2 = []
    out_contact_distance = []

    # 对距离进行排序，从全局最短的链接开始
    ind = np.argsort(contact_distance)
    for cur_pair_id in ind:
        pt1_id = contact_pts1[cur_pair_id]
        pt2_id = contact_pts2[cur_pair_id]
        if pt1_id not in out_contact_pts1 and pt2_id not in out_contact_pts2:
            out_contact_pts1.append(pt1_id)
            out_contact_pts2.append(pt2_id)
            out_contact_distance.append(contact_distance[cur_pair_id])

    return out_contact_pts1, out_contact_pts2, out_contact_distance


if __name__ == '__main__':
    pts1 = [[1,1], [5, 5], [100, 100], [150, 100]]
    pts2 = [[3,3], [70, 90], [140, 180]]
    pair_pts1_id, pair_pts2_id, dists = get_shortest_link_pair(pts1, pts2, 100)
    for i, j, d in zip(pair_pts1_id, pair_pts2_id, dists):
        print(i, j, d)
