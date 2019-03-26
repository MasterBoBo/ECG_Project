# -*- coding: utf-8 -*

"""

这是实现心电图智能识别的主函数，主要完成心电图的特征提取功能

"""

# TODO:这是最终的批量处理图片的文件 一定要好好写

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as ip
from PIL import Image
from PIL import ImageFilter
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sympy import *

import Project.ECG_FeatureExtract.QRS_Detector as qrs
import Project.Utils.Utils as Ut

'''
###########程序用到的参数##########
'''
eps: int = 100  # R波检测进行DBScan的epsilon
minPts: int = 10  # R波检测进行DBScan的MinPoints
threshold = 170  # RGB值的阈值
path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy/'  # 此处是心电图的所在文件夹
passBand = 0.08  # Passband edge frequencies.
stopBand = 0.8  # Stopband edge frequencies.
passGain = 1  # The maximum loss in the passband (dB).
stopGain = 2  # The minimum attenuation in the stopband (dB).
# passBand = 0.005  # Passband edge frequencies.
# stopBand = 0.3  # Stopband edge frequencies.
# passGain = 3  # The maximum loss in the passband (dB).
# stopGain = 5  # The minimum attenuation in the stopband (dB).
'''
#################################
'''

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
    box = (88, 1904, 3158, 2080)  # 7-1.jpg
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
    img_gray = image_enhanced.convert('L')
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
        y[i] = (-y[i] + 126) * (1 / 117.5)  # Fixme: 可能是由于循环的顺序问题此处画出的图是上下颠倒的，需要用-j来暂时修正。后期研究完善此处的坐标倒置问题
    for i in range(len(x)):
        x[i] = (x[i] - 119) * (10000 / 2954)

        # R波识别检测
        # 从第二个R波波峰到第三个R波波峰截取一段粗来分析
    ################################################################################################################
    # 获得原始图片
    XY = qrs.data_format(x, y)

    clustered = qrs.cluster(XY, eps=eps, MinPts=minPts)
    separated = qrs.separator(clustered)
    r_points = qrs.R_summit_detector(XY, separated)

    XY_New = qrs.x_y_to_XY(x, y)
    XY_Res = []
    for i in XY_New:
        if r_points[1, 0] < i[0] < r_points[2, 0]:
            XY_Res.append(i)
    XY_Res = np.array(XY_Res)
    plt.scatter(XY_Res[:, 0], XY_Res[:, 1], color='black', marker='.', linewidths=0.1, label='Original Pixels')
    plt.xlabel('Time (ms)')

    plt.ylabel('Voltage (mV)')
    # plt.scatter(XY_Res[390:415,0],XY_Res[390:415,1],color="red")
    # plt.scatter(XY_Res[550:565, 0], XY_Res[550:565, 1], color="red")
    plt.title('Original Pixels (one period starting from the summit of R wave)')
    # plt.show()

    #################################################################################################################
    # 获取巴特沃斯滤波器滤波之后的图像
    x_array = XY_Res[:, 0]
    y_array = XY_Res[:, 1]
    Ord, Wn = signal.buttord(passBand, stopBand, passGain, stopGain)
    b, a = signal.butter(Ord, Wn, output='ba')
    y_filtered = signal.filtfilt(b, a, y_array)
    plt.title('Interpolation')

    plt.scatter(x_array, y_filtered, color='green', marker='.', linewidths=0.1, label='Butterworth Filtered')
    plt.title('Butterworth Filtered')
    # plt.show()

    #################################################################################################################
    fx = ip.interp1d(x_array, y_filtered, 'slinear')
    plt.plot(x_array, fx(x_array), c='r', label='Interpolation')

    plt.legend(loc='upper center', fontsize=8)
    XY_New = np.array(XY_New)
    s_points = qrs.s_summit_detector(XY_New, r_points)
    s_points = np.array(s_points)
    # plt.scatter(s_points[:,0],s_points[:,1], color='red')
    # plt.show()
    ############################################################
    x1 = list(x_array)
    y1 = list(y_array)
    newx, newy = Ut.line_thinner(x1, y1)

    # newx = x
    # newy = y
    x_temp = np.array(newx)
    y_temp = np.array(newy)
    x_array = Ut.MaxMinNorm(x_temp)
    y_array = Ut.MaxMinNorm(y_temp)

    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0)
    x_train, y_train = Ut.sortXY(x_train, y_train)
    x_test, y_test = Ut.sortXY(x_test, y_test)
    x_test = np.array(x_test)
    x_train = np.array(x_train)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    rmses = []
    degrees = np.arange(1, 80)
    min_rmse, min_deg, score = 1e10, 0, 0
    deg = 30
    # for deg in degrees:
    # 生成多项式特征集 （如：根据degree = 3， 生成[[x,x**2,x**3]])
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    x_train_poly = poly.fit_transform(x_train)

    # 多项式拟合
    poly_reg = LinearRegression()
    poly_reg.fit(x_train_poly, y_train)
    # print(poly_reg.coef_,poly_reg.intercept_) # 系数及常数

    # 测试集比较
    x_test_poly = poly.fit_transform(x_test)
    y_test_pred = poly_reg.predict(x_test_poly)

    # mean_squared_error(y_true, y_pred) # 均方误差回归损失，越小越好
    poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmses.append(poly_rmse)

    # r2 范围[0, 1], R2越接近1拟合越好。
    r2score = r2_score(y_test, y_test_pred)

    # degree交叉验证
    error_result_file = open('/Users/brian/Desktop/结果/pic-6(9-11)/results.txt', mode='w')

    for deg in degrees:
        # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
        poly = PolynomialFeatures(degree=deg, include_bias=True)
        x_train_poly = poly.fit_transform(x_train)

        # 多项式拟合
        poly_reg = LinearRegression()
        poly_reg.fit(x_train_poly, y_train)

        print(poly_reg.coef_, poly_reg.intercept_)  # 系数及常数

        # 测试集比较
        x_test_poly = poly.fit_transform(x_test)
        y_test_pred = poly_reg.predict(x_test_poly)

        # mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
        poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmses.append(poly_rmse)
        # r2 范围[0，1]，R2越接近1拟合越好。
        r2score = r2_score(y_test, y_test_pred)

        # degree交叉验证
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg
            score = r2score
        # error_result_file.write('degree = %s, RMSE = %.2f ,r2_score = %.2f /' % (deg, poly_rmse, r2score))
        print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse, r2score))
    error_result_file.close()
    # img_gray.show()
    fig1 = plt.figure()
    plt.xlabel('Time (ms)')
    plt.title('Polynomial Fit')
    plt.ylabel('Voltage (mV)')
    plt.plot(x_train, y_train, '.', color='red', label='Original')
    plt.plot(x_test, y_test_pred, color='green', label='Polynomial Fit')
    plt.legend(loc='upper center', fontsize=8)
    plt.show()
