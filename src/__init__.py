#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:__init__.py.py
@TIME:2018/12/5 15:43
@DES:


'''

from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D,Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout,Activation

from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
import cv2
import keras


from config import  config
def build_model03(config):
    model = Sequential()
    inputShape = (config.normal_size, config.normal_size, config.channels)
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def build_leNet(config):
    # initialize the model
    model = Sequential()
    inputShape = (config.normal_size, config.normal_size, config.channels)
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(config.classes))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # return the constructed network architecture
    return model

def build_AlexNet(config):
    model = Sequential()
    input_shape = (config.normal_size, config.normal_size, config.channels)
    model.add(Convolution2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(
        Convolution2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    print( "model is ready！")
    return model


def train(model,config,train_x,train_y,dev_x,dev_y):
    lr_reducer = ReduceLROnPlateau(factor=0.005, cooldown=0, patience=5, min_lr=0.5e-6,verbose=1)      #设置学习率衰减
    early_stopper = EarlyStopping(min_delta=0.001, patience=10,verbose=1)                                     #设置早停参数
    checkpoint = ModelCheckpoint(config.weights_path + config.model_name + "_model.h5",
                                 monitor="val_acc", verbose=1,
                                 save_best_only=True, save_weights_only=True,mode="max")            #保存训练过程中，在验证集上效果最好的模型
    #使用数据增强
    if config.data_augmentation:
        print("using data augmentation method")
        data_aug = ImageDataGenerator(
            rotation_range=90,              #图像旋转的角度
            width_shift_range=0.2,          #左右平移参数
            height_shift_range=0.2,         #上下平移参数
            zoom_range=0.3,                 #随机放大或者缩小
            horizontal_flip=True,           #随机翻转
        )
        data_aug.fit(train_x)
        model.fit_generator(
            data_aug.flow(train_x,train_y,batch_size=config.batch_size),
            steps_per_epoch=train_x.shape[0] // config.batch_size,
            validation_data=(dev_x,dev_y),
            shuffle=True,
            epochs=config.epochs,verbose=1,max_queue_size=100,
            callbacks=[lr_reducer,early_stopper,checkpoint]
        )
    else:
        print("don't use data augmentation method")
        model.fit(train_x,train_y,batch_size = config.batch_size,
                  nb_epoch=config.epochs,
                  validation_data=(dev_x, dev_y),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, checkpoint]
                  )
        model.save(config.model_name+'.h5')


def get_datas(config):
    type_list = os.listdir(config.path_root)
    type_list.sort()
    '''声明空的训练样本对'''
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    dictionary_typename_imgnum = {}
    dictionary_typename_lable = {}

    num_of_type = len(type_list)
    # print "hello"
    k = 0
    while(k<config.classes):
    # while(k<len(type_list)):
        type_name = type_list[k]
        print ('current type is :' + type_name)
        dictionary_typename_lable[type_name] = k
        # dictionary_typename_lable[type_name] = (k+1)/num_of_type  #每一个类型都用一个不大于1的数来表示
        path_typedir = config.path_root+'/'+type_name
        image_list = os.listdir(path_typedir)
        num_of_img = len(image_list)
        # dictionary_typename_imgnum[type_name]=num_of_img  #将类型名称和 样本数量存入字典中

        '''接下来划分训练集和测试集'''
        cut_point = int(num_of_img*(config.cut_rate))
        for image in image_list[0:cut_point]:
            img_name = image
            # print ("handling "+img_name)
            path_img = config.path_root+"/"+type_list[k]+'/'+img_name
            img_value = Image.open(path_img)
            '''判断是否为三通道'''
            if img_value.mode != "RGB":
                # print (img_name+"is not 3 dims")
                # dictionary_typename_imgnum[type_name]=dictionary_typename_imgnum[type_name]-1
                continue
            img_value = cv2.imread(path_img)
            img_value = cv2.resize(img_value,(config.normal_size,config.normal_size))
            img_value = img_to_array(img_value)

            # img_value = img_value.resize((config.normal_size,config.normal_size),Image.ANTIALIAS)
            # img_value = np.array(img_value)
            x_train.append(img_value)
            y_train.append(dictionary_typename_lable[type_name])
        for image in image_list[cut_point:-1]:
            img_name = image
            # print ("handling " + img_name)
            path_img = config.path_root +'/'+type_list[k]+'/'+img_name
            img_value = Image.open(path_img)
            '''判断是否为三通道'''
            if img_value.mode != "RGB":
                # print (img_name + "is not 3 dims")
                # dictionary_typename_imgnum[type_name] = dictionary_typename_imgnum[type_name] - 1
                continue
            img_value = cv2.imread(path_img)
            img_value = cv2.resize(img_value, (config.normal_size, config.normal_size))
            img_value = img_to_array(img_value)

            # img_value = img_value.resize((config.normal_size,config.normal_size),Image.ANTIALIAS)
            # img_value = np.array(img_value)

            x_test.append(img_value)
            y_test.append(dictionary_typename_lable[type_name])
        k = k+1
    # npx_train = np.array(x_train).reshape(-1,config.normal_size,config.normal_size,3)
    # # npy_train = to_categorical(np.array(y_train))
    # npy_train = np.array(y_train)
    # npy_train = myone_hot(npy_train)
    # npx_test = np.array(x_test).reshape(-1,config.normal_size,config.normal_size,3)
    # npy_test = np.array(y_test)
    # npy_test = myone_hot(npy_test)
    x_train = np.array(x_train,dtype="float")/255.0
    x_test= np.array(x_test,dtype="float")/255.0
    y_train = myone_hot(np.array(y_train))
    y_test = myone_hot(np.array(y_test))
    '''此时不做处理，只返回数据'''

    return x_train,y_train,x_test,y_test


def get_datas02(config):
    type_list = os.listdir(config.path_root)
    type_list.sort()
    '''声明空的训练样本对'''
    x=[]
    y=[]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    dictionary_typename_lable = {}
    num_of_type = len(type_list)
    # print "hello"
    k = 0
    while(k<config.classes):
    # while(k<len(type_list)):
        type_name = type_list[k]
        print ('current type is :' + type_name)
        dictionary_typename_lable[type_name] = k
        # dictionary_typename_lable[type_name] = (k+1)/num_of_type  #每一个类型都用一个不大于1的数来表示
        path_typedir = config.path_root+'/'+type_name
        image_list = os.listdir(path_typedir)
        num_of_img = len(image_list)
        for image in image_list:  #全体做遍历
            img_name = image
            path_img = config.path_root + "/" + type_list[k] + '/' + img_name
            img_value = Image.open(path_img)
            '''判断是否为三通道'''
            if img_value.mode != "RGB":
                # print (img_name+"is not 3 dims")
                # dictionary_typename_imgnum[type_name]=dictionary_typename_imgnum[type_name]-1
                continue
            img_value = cv2.imread(path_img)
            img_value = cv2.resize(img_value, (config.normal_size, config.normal_size))
            img_value = img_to_array(img_value)
            x.append(img_value)
            y.append(dictionary_typename_lable[type_name])
        k = k + 1

    '''接下来划分训练集和测试集'''
    cut_point = int(num_of_img*(config.cut_rate))
    y = myone_hot(np.array(y))
    x = np.array(x,dtype ="float")/255.0

    x_train = x[0:cut_point]
    x_test = x[cut_point:-1]
    y_train = y[0:cut_point]
    y_test = y[cut_point:-1]

    '''此时不做处理，只返回数据'''

    return x_train,y_train,x_test,y_test

# def get_datas03(config):
#     type_list = os.listdir(config.path_root)
#     type_list.sort()
#     '''声明空的训练样本对'''
#     x=[]
#     y=[]
#     x_train = []
#     y_train = []
#     x_test = []
#     y_test = []
#     dictionary_typename_lable = {}
#     num_of_type = len(type_list)
#     # print "hello"
#     load_num = 0
#     k=0
#     while(k<config.classes):
#     # while(k<len(type_list)):
#         type_name = type_list[k]
#         print ('current type is :' + type_name)
#         dictionary_typename_lable[type_name] = k
#         # dictionary_typename_lable[type_name] = (k+1)/num_of_type  #每一个类型都用一个不大于1的数来表示
#         path_typedir = config.path_root+'/'+type_name
#         image_list = os.listdir(path_typedir)
#         num_of_img = len(image_list)
#         index_img = current_img
#         while(index_img<num_of_img):
#         # for image in image_list:  #全体做遍历
#             if(load_num==config.batch_size):
#                 end_flag=1
#                 break
#             img_name = image_list[index_img]
#             path_img = config.path_root + "/" + type_list[k] + '/' + img_name
#             img_value = Image.open(path_img)
#             '''判断是否为三通道'''
#             if img_value.mode != "RGB":
#                 # print (img_name+"is not 3 dims")
#                 # dictionary_typename_imgnum[type_name]=dictionary_typename_imgnum[type_name]-1
#                 continue
#             img_value = cv2.imread(path_img)
#             img_value = cv2.resize(img_value, (config.normal_size, config.normal_size))
#             img_value = img_to_array(img_value)
#             x.append(img_value)
#             y.append(dictionary_typename_lable[type_name])
#             load_num+=1
#             index_img +=1
#         k = k + 1
#
#     '''接下来划分训练集和测试集'''
#     cut_point = int(num_of_img*(config.cut_rate))
#     y = myone_hot(np.array(y))
#     x = np.array(x,dtype ="float")/255.0
#
#     x_train = x[0:cut_point]
#     x_test = x[cut_point:-1]
#     y_train = y[0:cut_point]
#     y_test = y[cut_point:-1]
#
#     '''此时不做处理，只返回数据'''
#
#     return x_train,y_train,x_test,y_test

def myone_hot(np_array):
    a = np_array
    n_class = a.max() + 1
    n_sample = a.shape[0]
    b = np.zeros((n_sample, n_class))  # 3个样本，4个类别
    b[:, a] = 1  # 非零列赋值为1
    b = np.array(b)
    return b

# def pre_handle(x_train,y_train,x_test,y_test):


import sys

from predict import  get_model
if __name__ == '__main__':
    # pre_handle()
    # model = build_leNet(config)
    model = build_AlexNet(config)
    # model = build_model(config)
    # model.load_weights("../log/LeNet01_model.h5")
    # model = get_model("../log/h5")
    # model = get_model("LeNet01.h5")
    train_x,train_y,dev_x,dev_y = get_datas02(config)


    # 将终端输出指向文件
    # log_file_name= config.model_name+'.txt'
    # output = sys.stdout
    # outputfile = open(log_file_name, 'w')
    # outputfile.write(" ")
    # sys.stdout = outputfile

    train(model,config,train_x,train_y,dev_x,dev_y)

    # # 将终端输出修改回来
    # outputfile.close()
    # sys.stdout = output

    # print testmodel.summary
