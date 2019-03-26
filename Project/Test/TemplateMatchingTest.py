import os

import cv2
import numpy as np

path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy/'
counter = 0
template_o = cv2.imread(path + "7-1.jpg")
template = template_o[1904:2081, 88:3160]

fileList = os.listdir(path)

fileList_temp = os.listdir(path)
for f in fileList_temp:  # 首先对文件夹内的文件做一个清洗，除去所有隐藏文件和非所需图像文件
    if not f.endswith('.jpg'):
        fileList.remove(f)

fileList = sorted(fileList)

for files in fileList:
    img = cv2.imread(path + files)

    h, w = template.shape[:2]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6

    loc = np.where(res >= threshold)
    print(loc)

    if loc[0].__len__() == 0:
        print("未识别")
        counter += 1
    else:
        print("已识别")

    for pt in zip(*loc[::-1]):
        right_bottom = (pt[0] + w, pt[1] + h)
        zero_point = (pt[0] + 119, pt[1] + 130)
        cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)
        cv2.circle(img, zero_point, 10, (0, 0, 255))
    cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
    cv2.imshow('input_image', img)
    cv2.imwrite(path + "results/" + files, img)

    print("共有" + str(len(fileList)) + "个文件，" + '未识别：' + \
          str(counter) + "个。识别率" + str(((len(fileList) - counter) /
                                        len(fileList)) * 100) + '%')
