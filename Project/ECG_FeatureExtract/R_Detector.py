from typing import List, Any

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.cluster import DBSCAN


def data_format(x: list, y: list) -> list:
    XY = []  # XY为要进行DBScan的所有坐标点，这里只保留X大于0的点,然后从上往下扫描直到扫描到有图像之后的10个像素。
    if len(x) == len(y):
        for i in range(len(x)):
            if x[i] > 0:
                if (y[i] > max(y[i:y.__len__()]) - 0.2) and (y[i] < max(y[1:y.__len__()])):
                    XY_Temp = []
                    XY_Temp.append(x[i])
                    XY_Temp.append(y[i])
                    XY.append(XY_Temp)
    else:
        print('X坐标与Y坐标的数量不一致')
    return XY


def cluster(XY: list, eps, MinPts) -> np.ndarray:
    """
    此函数对满足条件（*）的坐标进行聚类，获得多个簇
    满足条件（*）：这里只保留X大于0的点,然后从上往下扫描直到扫描到有图像之后的10个像素。
    :param MinPts: int
    :param eps: int
    :param XY : list
    :rtype np.ndarray
    """

    clustered: np.ndarray = DBSCAN(eps=eps, min_samples=MinPts).fit_predict(XY)
    return clustered


def separator(clustered: np.ndarray) -> list:
    """
    **这是一个用来将DBScan聚类之后的多个簇分开的函数，由于不了解SKLearn包内DBScan的函数
    怎么返回每个簇的结果只能这样人工来做每个簇的分开。。。。**
    :rtype: list of lists
    """
    clusters: List[List[Any]] = []
    cluster_temp = []
    cluster_counter: int = 0
    for i in range(len(clustered)):
        if not clustered[i] < 0:
            if cluster_counter == clustered[i]:
                cluster_temp.append(i)
            elif clustered[i] == (cluster_counter + 1):
                clusters.append(cluster_temp)
                cluster_temp = []
                cluster_temp.append(i)
                cluster_counter += 1
    clusters.append(cluster_temp)
    return clusters


def R_points_detector(XY: list, clusters: list) -> np.ndarray:
    r_points = []
    for i in range(len(clusters)):
        compare_ring = []
        for j in range(len(clusters[i])):
            compare_ring.append(XY[clusters[i][j]][1])
        idx = compare_ring.index(max(compare_ring))
        r_points.append(XY[clusters[i][idx]])
    Rs: ndarray = np.array(r_points)
    return Rs
