# -*- coding: utf-8 -*

"""

这是实现心电图智能识别的主函数，主要完成心电图的特征提取功能

"""

# TODO:这是最终的批量处理图片的文件 一定要好好写

import os

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

from Project.ECG_FeatureExtract.R_Detector import *

###########程序用到的参数##########
eps: int = 100  # R波检测进行DBScan的epsilon
minPts: int = 10  # R波检测进行DBScan的MinPoints
threshold = 170  # RGB值的阈值
path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy/'  # 此处是心电图的所在文件夹
#################################

fileList = os.listdir(path)
fileList_temp = os.listdir(path)

for f in fileList_temp:  # 首先对文件夹内的文件做一个清洗，除去所有隐藏文件和非所需图像文件
    if not f.endswith('.jpg'):
        fileList.remove(f)

for files in fileList:
    image = Image.open(path + files)
    size = image.size
    print(size)

    if size == (3189, 2362):
        picQuan = 'high'
    elif size == (1000, 740):
        picQuan = 'low'

    if picQuan == 'high':  # Fixme: 此处切出的第二导联图像有问题，不同的心电图第二导联的坐标稍有不同，后期需要研究使用智能切割方法切出第二导联图像
        box = (89, 1923, 3160, 2180)  # 切出高分辨率第二导联的图像
    elif picQuan == 'low':
        box = (26, 609, 990, 690)  # 切出低分辨率第二导联的图像
        # image_enhanced ==> img
    img = image.crop(box)  # img是切出的第二导联图像
    box_size = img.size
    width: int = box_size[0]
    height: int = box_size[1]
    pixel = img.load()

    # 将小于阈值的像素转换成黑色，其他的像素转换成白色。
    i: int
    for i in range(0, width):
        for j in range(0, height):

            if threshold > pixel[i, j][0] and pixel[i, j][1] < threshold and pixel[i, j][2] < threshold:
                pixel[i, j] = (255, 255, 255, 255)
            else:
                pixel[i, j] = (0, 0, 0, 255)
    # img ==> img_sharpened
    image_sharpened = img.filter(ImageFilter.SHARPEN)
    image_detailed = image_sharpened.filter(ImageFilter.DETAIL)
    image_enhanced = image_detailed.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # image_enhanced ==> image_gray
    img_gray = img.convert('L')
    img_gray_pixel = img_gray.load()
    x = []
    y = []
    for i in range(0, width):
        for j in range(0, height):
            if img_gray_pixel[i, j] != 0:
                x.append(i)
                y.append(j)

    # 将原点与图片上的原点对齐，矫正X，Y坐标。
    for i in range(len(y)):
        y[i] = (-y[i] + 194.5) * (1 / 117.5)  # Fixme: 可能是由于循环的顺序问题此处画出的图是上下颠倒的，需要用-j来暂时修正。后期研究完善此处的坐标倒置问题
    for i in range(len(x)):
        x[i] = (x[i] - 108) * (10000 / 2954)

    # R波识别检测
    XY = data_format(x, y)
    clustered = cluster(XY, eps=eps, MinPts=minPts)
    separated = separator(clustered)
    r_points = R_points_detector(XY, separated)

    plt.scatter(x, y, marker='.', s=1)
    plt.scatter(r_points[:, 0], r_points[:, 1], marker='+', s=100)
    plt.show()
