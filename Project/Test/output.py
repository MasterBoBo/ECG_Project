import os
from typing import List

import pandas as pd
from PIL import Image
from PIL import ImageFilter

path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy/'
threshold = 170

fileList: List[str] = os.listdir(path)
fileList_temp = os.listdir(path)

for f in fileList_temp:  # 首先对文件夹内的文件做一个清洗，除去所有隐藏文件和非所需图像文件
    if not f.endswith('.jpg'):
        fileList.remove(f)

fileList = sorted(fileList)

for files in fileList:
    image = Image.open(path + files)
    size = image.size

    if size == (3189, 2362):
        picQuan = 'high'
    elif size == (1000, 740):
        picQuan = 'low'

    # if picQuan == 'high':  # Fixme: 此处切出的第二导联图像有问题，不同的心电图第二导联的坐标稍有不同，后期需要研究使用智能切割方法切出第二导联图像
    #     box = (89, 1923, 3160, 2180)  # 切出高分辨率第二导联的图像
    # elif picQuan == 'low':
    #     box = (26, 609, 990, 690)  # 切出低分辨率第二导联的图像

    # box = (284, 1995, 510, 2143)
    box = (84, 1900, 3170, 2080)

    img = image.crop(box)

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
    img_gray = image_enhanced.convert('L')
    img_gray_pixel = img_gray.load()
    x = []
    y = []
    for i in range(0, width):
        for j in range(0, height):
            if img_gray_pixel[i, j] != 0:
                x.append(i)
                y.append(j)
    img_gray.show()
    # 将原点与图片上的原点对齐，矫正X，Y坐标。
    for i in range(len(y)):
        y[i] = (-y[i] + 194.5) * (1 / 117.5)  # Fixme: 可能是由于循环的顺序问题此处画出的图是上下颠倒的，需要用-j来暂时修正。后期研究完善此处的坐标倒置问题
    for i in range(len(x)):
        x[i] = (x[i] - 108) * (10000 / 2954)

    Data = {
        'X': x,
        'Y': y,
    }

    RES = pd.DataFrame(Data)
    RES.to_csv(path + 'results/' + 'Results' + '-' + files + '.csv', encoding='gbk', index=False)
