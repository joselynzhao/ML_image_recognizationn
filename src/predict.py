#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:predict.py
@TIME:2018/12/7 16:27
@DES:
'''

import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from config import config
from keras.models import load_model
from config import config
# from utils import build_model

def get_model(model_file_name):
    try:
        import h5py

        # print ('import fine')
    except ImportError:
        h5py = None

    model = load_model(model_file_name)
    print str(model_file_name)+" 加载成功~"
    # model.summary()
    return model

def predict(config):
    weights_path = config.weights_path + config.model_name + "_model.h5"
    data_path = config.test_data_path
    data_list = os.listdir(data_path)
    data = []
    for file in data_list:
        file_name = data_path + file
        image = cv2.imread(file_name)
        image = cv2.resize(image, (config.normal_size,config.normal_size))
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float") / 255.0

    # model = build_model(config)
    # model.load_weights(weights_path)
    model = get_model("output_model.h5")
    pred = model.predict(data)
    print(pred)


import  random
def mypredict(config):
    which_kind = random.randint(0, 101)
    type_list = os.listdir(config.path_root)
    type_name = type_list[which_kind]
    path_imgs = config.path_root+'/'+ type_list[which_kind]
    img_list = os.listdir(path_imgs)
    which_img = random.randint(0,len(img_list))
    img_name = img_list[which_img]
    path_img = config.path_root+'/'+type_name+'/'+img_name
    image = cv2.imread(path_img)
    image = cv2.resize(image,(config.normal_size,config.normal_size))
    image = img_to_array(image)
    data = []
    data.append(image)
    data = np.array(image,dtype="float")/255.0
    model = get_model("output_model.h5")
    pred = model.predict(data)
    print (pred)

    print "hello"

def mypredict02(config):
    which_kind = random.randint(0, 101)
    type_list = os.listdir(config.path_root)
    type_name = type_list[which_kind]
    path_imgs = config.path_root+'/'+ type_list[which_kind]
    img_list = os.listdir(path_imgs)
    data =  []
    for img_name in img_list:
        path_img = config.path_root + '/' + type_name + '/' + img_name
        image = cv2.imread(path_img)
        image = cv2.resize(image, (config.normal_size, config.normal_size))
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float")/255.0

    model = get_model("output_model.h5")
    pred = model.predict(data)
    print (pred)

    print "hello"

if __name__ =="__main__":
    mypredict02(config)
    # predict(config)