# -*- coding: utf-8 -*

"""

这是识别心电图图片上文字的程序，大部分内容利用百度人工智能识别接口，需要联网。
诊断内容部分采用Tesseract识别，无需联网

"""

# FIXME：还未实现Tesseract识别

import base64
import os
import re

import pandas as pd
import requests
from PIL import Image
from pytesseract import pytesseract

from Project.ECG_OCR.doAuth import auth

Path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy/'  # FIXME: 这是计算机储存所有ECG图片的位置，在运行前更换为自己的路径
access_token = auth()  # 获取token
url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=' + access_token
fileList = os.listdir(Path)

# 以下开始批量处理图片

nameList = []
sexList = []
ageList = []
conclusionList = []
PRList = []
HRList = []
axisList = []
qrsList = []
rv5List = []
qtList = []
sv1List = []
qtcList = []
rsList = []
indexList = []
fileTotal = 0

for files in fileList:
    if files.endswith('.jpg'):
        fileTotal = fileTotal + 1
    else:
        fileList.remove(files)

# TODO: 此处插入一个doSort排序函数用来给fileList排序

counter = 0
for files in fileList:
    counter += 1
    fileName = files.split('.', 2)[0]
    indexList.append(fileName)

    print('共' + str(fileList.__len__()) + '个文件，正在处理第' + str(counter) + '个。已完成' +
          '正在处理' + files + ' ' + '请勿关闭' + '\n' +
          str((counter / fileList.__len__()) * 100) + '%')
    image = Image.open(Path + files)
    size = image.size
    if size == (3189, 2362):
        picQuan = 'high'
    elif size == (1000, 740):
        picQuan = 'low'

    if picQuan == 'high':
        box = (1750, 150, 3150, 500)  # 切出高分辨率疾病诊断部分的图像
    elif picQuan == 'low':
        box = (26, 609, 990, 690)  # 切出低分辨率疾病诊断部分的图像

    concluSection = image.crop(box)
    concluSection.show()
    conclusion = pytesseract.image_to_string(concluSection, lang='chi_sim')  # Tesseract处理诊断结论部分
    print(conclusion)
    conclusionList.append(conclusion)

    ##########################以下是调用百度接口的代码 处理其他数据的识别############################
    f = open(Path + files, 'rb')
    # templateSign = 'ad48b9dd296af3d87c3a8d75f1b5de4a'  # 参数templateSign 百度自定义模版识别的模版ID
    img = base64.b64encode(f.read())  # 参数image：图像base64编码
    params = {"image": img}
    # "templateSign": templateSign}

    headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
    request = requests.post(url, params, headers=headers)
    result = request.json()

    print(result)
    content = str(result)
    f.close()

    if content:
        print(content)

        result_name = re.findall(".*名:(.*?)'}", content)  # 在返回的结果中匹配寻找姓名
        if result_name == []:
            nameList.append(' ')
        else:
            nameList.append(result_name[0])  # 返回的结果加入姓名列表

        result_sex = re.findall(".*性别:(.*?)'}", content)  # 在返回的结果中匹配寻找姓名
        if result_sex == []:
            sexList.append(' ')
        else:
            sexList.append(result_sex[0])

        result_age = re.findall(".*年龄:(.*?)岁", content)  # 在返回的结果中匹配寻找姓名
        if result_age == []:
            ageList.append(' ')
        else:
            ageList.append(result_age[0])

        # result_conclusion = re.findall(".*心电图诊断:(.*?)'}", content)  # 在返回的结果中匹配寻找姓名
        # if result_conclusion == []:
        #     conclusionList.append(' ')
        # else:
        #     conclusionList.append(result_conclusion[0])

        result_PR = re.findall(".*P-R:(.*?)毫秒", content)  # 在返回的结果中匹配寻找姓名
        if result_PR == []:
            PRList.append(' ')
        else:
            PRList.append(result_PR[0])

        result_HR = re.findall(".*HR:(.*?)次/分", content)  # 在返回的结果中匹配寻找姓名
        if result_HR == []:
            HRList.append(' ')
        else:
            HRList.append(result_HR[0])

        result_axis = re.findall(".*电轴:(.*?)'}", content)  # 在返回的结果中匹配寻找姓名
        if result_axis == []:
            axisList.append(' ')
        else:
            axisList.append(result_axis[0])

        result_qrs = re.findall(".*QRS:(.*?)毫秒", content)  # 在返回的结果中匹配寻找姓名
        if result_qrs == []:
            qrsList.append(' ')
        else:
            qrsList.append(result_qrs[0])

        result_rv5 = re.findall(".*Rv5:(.*?)毫伏", content)  # 在返回的结果中匹配寻找姓名
        if result_rv5 == []:
            rv5List.append(' ')
        else:
            rv5List.append(result_rv5[0])

        result_qt = re.findall(".*Q-T:(.*?)毫秒", content)  # 在返回的结果中匹配寻找姓名
        if result_qt == []:
            qtList.append(' ')
        else:
            qtList.append(result_qt[0])

        result_sv1 = re.findall(".*Sv1:(.*?)毫伏", content)  # 在返回的结果中匹配寻找姓名
        if result_sv1 == []:
            sv1List.append(' ')
        else:
            sv1List.append(result_sv1[0])

        result_qtc = re.findall(".*QTc:(.*?)'}", content)  # 在返回的结果中匹配寻找姓名
        if result_qtc == []:
            qtcList.append(' ')
        else:
            qtcList.append(result_qtc[0])

        result_rs = re.findall(".*R\+S:(.*?)毫伏", content)  # 在返回的结果中匹配寻找姓名
        if result_rs == []:
            rsList.append(' ')
        else:
            rsList.append(result_rs[0])

        print('匹配完毕')
    else:
        print('图像' + files + '未识别出结果')

Data = {'编号': indexList,
        '姓名': nameList,
        '性别': sexList,
        '年龄': ageList,
        '心电图诊断': conclusionList,
        'P-R': PRList,
        'HR': HRList,
        '电轴': axisList,
        'QRS': qrsList,
        'RV5': rv5List,
        'Q-T': qtList,
        'Sv1': sv1List,
        'R+S': rsList,
        'QTc': qtcList}

print('Data 装载完毕')
print(Data)
Res = pd.DataFrame(Data)
print('Res准备好')
print(Res)

Res.to_csv(Path + 'results/' + 'Results.csv', encoding='gbk', index=False)
