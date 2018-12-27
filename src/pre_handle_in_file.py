#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:pre_handle_in_file.py
@TIME:2018/12/13 11:44
@DES:
'''

import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
import cv2
import keras
from config import config

train_set_file_name = "train_set.txt"
test_set_file_name ="test_set.txt"

lable_to_number = "lable.txt"



def handle_data(config):
    file_train = open(train_set_file_name, "w")
    file_test = open(test_set_file_name, "w")
    file_label = open(lable_to_number,"w")
    type_list=os.listdir(config.path_root)
    for type in type_list:
        file_label.write(type+"\n")
        image_list = os.listdir(config.path_root+'/'+type)
        num_of_image = len(image_list)
        cut_point = int(num_of_image*(config.cut_rate))
        point = cut_point
        for k in range(cut_point):
            image_name = image_list[k]
            image_path = config.path_root+'/'+type+'/'+image_name
            '''只写3通道的图片'''
            image = Image.open(image_path)
            if image.mode != "RGB":
                continue
            file_train.write(image_path+" "+type+"\n")
        while point<num_of_image:
            image_name = image_list[point]
            image_path = config.path_root+'/'+type+'/'+image_name
            '''只写3通道的图片'''
            image = Image.open(image_path)
            if image.mode != "RGB":
                point=point +1
                continue
            file_test.write(image_path+" "+type+"\n")
            point = point +1

    file_test.close()
    file_train.close()
    file_label.close()



if __name__ =="__main__":
    handle_data(config)






