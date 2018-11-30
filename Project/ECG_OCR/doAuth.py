# -*- coding: utf-8 -*

"""

这是百度智能图片识别API的鉴权程序，用来return access_token

"""

import requests


def auth():
    # 此函数为百度智能图片识别的鉴权模块
    auth_key = 'Pjh4hlkpRGTajlWvHE1NqOao'  # 官网获取的AK
    client_key = 'QT8T5NFadyOXGqis3aid2voTHa5NtdKZ'  # 官网获取的CK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&' \
           'client_id=' + auth_key + '&client_secret=' + client_key  # 鉴权的URL
    headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
    request = requests.post(host, headers=headers)  # 使用Post方式
    response = request.json()
    result = response['access_token']

    if response:
        print(response)
        print('Token获取成功，本次的Token为：' + result)
        return result
    else:
        print('Token获取失败' + result['error'] + result['error_description'])
        return None
