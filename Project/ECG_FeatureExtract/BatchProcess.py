# -*- coding: utf-8 -*

"""

这是实现心电图智能识别的主函数，主要完成心电图的特征提取功能

"""

# TODO:这是最终的批量处理图片的文件 一定要好好写

import os

import matplotlib.pyplot as plt
from PIL import Image

path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy/'  # Fixme: 此处是心电图的所在文件夹

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

    if picQuan == 'high':
        box = (89, 1923, 3160, 2180)  # 切出高分辨率第二导联的图像
    elif picQuan == 'low':
        box = (26, 609, 990, 690)  # 切出低分辨率第二导联的图像
    img = image.crop(box)
    img_gray = img.convert('L')
    pix = image.load()
    width = size[0]
    height = size[1]
    plt.imshow(img, cmap=plt.get_cmap("gray"), vmin=100, vmax=150)

    # for x in range(width):
    #     for y in range(height):
    #         r, g, b = pix[x, y]
