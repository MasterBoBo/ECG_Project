# -*- coding: utf-8 -*
"""

这里的代码是把各种处理图像时候的小工具

"""
import numpy as np
import pandas as pd


def MaxMinNorm(inp: np.ndarray) -> np.ndarray:
    minium = inp.min()
    diff = inp.max() - minium
    res = []
    for i in inp:
        res.append((i - minium) / diff)
    res = np.array(res)

    return res


def line_thinner(x: list, y: list) -> tuple:
    """
    :type x: list
    :param x: list
    :param y: list
    :return: newX:list, newY:list
    """
    i = 0
    counter = 0
    newX: list = []
    newY: list = []
    while i < (len(x) - 1):
        if x[i] == x[i + 1]:
            counter += 1
        else:
            tempY = y[(i - counter):(i + 1)]
            newX.append(x[i])
            newY.append(max(tempY))
            tempY.clear()
            counter = 0
        i += 1
    return newX, newY


def sortXY(x: np.ndarray, y: np.ndarray) -> tuple:
    temp = pd.DataFrame([x, y]).T
    temp = temp.sort_values(0)
    resX = temp[0]
    resY = temp[1]
    return resX, resY
