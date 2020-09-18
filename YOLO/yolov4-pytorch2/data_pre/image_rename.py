# coding=utf-8
"""
修改一个文件夹下所有图片的名字，修改成 000000.jpg 格式
保存修改的顺序到一个文档中，方便以后查看
@author: libo
"""

import os
path = "./image/"
filelist = os.listdir(path)

for file in filelist:
    print(file)

count = 0     # 通过修改count得到图片名字的起始格式
for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]
    Newdir = os.path.join(path, str(count).zfill(6) + filetype)
    os.rename(Olddir, Newdir)
    count += 1

