# -*- coding: utf-8 -*

"""

这是处理从客户处得到的原始文件的程序，自动把得到的文件解压，自动命名

"""

# FIXME：目前只是实现了原始的重命名功能，需要完善，最好可以用界面编程，鼠标选择文件 然后自动解压文件。

import os

Path = '/Users/brian/MyProjects/ECG_Project/yyyyyyy'
f = os.listdir(Path)
print(f)
n = 0
for i in f:
    oldName = Path + f[n]
    newName = Path + '12-' + str(n + 1) + '.jpg'
    os.renames(oldName, newName)
    n += 1
