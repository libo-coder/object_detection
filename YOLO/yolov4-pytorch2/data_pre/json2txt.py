# coding=utf-8
"""
ImageSets/Main中的四个 TXT 文件形成
"""
import os
import random

# ==================可能需要修改的地方=====================================#
g_root_path = "../VOCdevkit_lb/VOC0807/"
jsonfilepath = "Annotations"  # 标注文件存放路径
saveBasePath = "ImageSets/Main/"  # ImageSets信息生成路径
trainval_percent = 1
train_percent = 1
# ==================可能需要修改的地方=====================================#

os.chdir(g_root_path)
total_json = os.listdir(jsonfilepath)
num = len(total_json)
json_list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(json_list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train  size", tr)
ftrainval = open(saveBasePath + "trainval.txt", "w")
ftest = open(saveBasePath + "test.txt", "w")
ftrain = open(saveBasePath + "train.txt", "w")
fval = open(saveBasePath + "val.txt", "w")

for i in json_list:
    name = total_json[i][:-5] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
