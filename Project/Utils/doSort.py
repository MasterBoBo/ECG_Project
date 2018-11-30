# -*- coding: utf-8 -*
"""

这是专门依据心电图图片名称编号来排序的函数
使用前请确保传入的List中心电图图片的名字都是：
batchNumber-indexNumber.jpg的形式
形如：1-3.jpg, 1-1.jpg, 2-10.jpg, ...

"""


def doSort(list):
    for i in list:
        i.split('.jpg')

        # TODO: 此处写fileList的排序函数，用来给图片的名字排序（形如：1-3.jpg, 1-1.jpg, 2-10.jpg, ...)
