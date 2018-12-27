#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:main01.py
@TIME:2018/12/13 15:09
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
from keras.layers.normalization import  BatchNormalization
from config import config
import  os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
import cv2
import keras
# from sklearn import preprocessing

from keras.utils import np_utils
from tensorflow import one_hot
from keras.preprocessing.image import img_to_array

def build_AlexNet(config):
    model = Sequential()
    input_shape = (config.normal_size, config.normal_size, config.channels)
    model.add(Convolution2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(
        Convolution2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
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


def next_batch(train_data, train_target, batch_size):
    index = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

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
                  nb_epoch=1,
                  validation_data=(dev_x, dev_y),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, checkpoint]
                  )
        model.save(config.model_name+'.h5')

def myone_hot(np_array):
    a = np_array
    n_class = config.classes+1
    n_sample = a.shape[0]
    b = np.zeros((n_sample, n_class))  # 3个样本，4个类别
    b[:, a] = 1  # 非零列赋值为1
    b = np.array(b)
    return b

def get_train_data(config,current_line,datas,labels):
    x = []
    y = []
    # print "正在处理训练数据格式。"
    # k = current_line
    for k in range(current_line,len(datas)):
    # while( k<len(datas)):
    #     print k,len(datas)
        if k==current_line+config.batch_size:
            break
        key_value = datas[k].split(" ")
        # image = Image.open(key_value[0])
        # '''"判断是否为3通道"'''
        # if image.mode !="RGB":
        #     continue
        image = cv2.imread(key_value[0])
        image = cv2.resize(image,(config.normal_size,config.normal_size))
        image = img_to_array(image)
        x.append(image)
        if key_value[1] in labels:
            y.append(labels.index(key_value[1]))
        else:
            print "没有找到标签！"

    x = np.array(x, dtype="float") / 255.0
    # y = myone_hot(np.array(y))
    # y = one_hot(np.array(y),config.classes,1,0)
    y = np_utils.to_categorical(y,config.classes)
    return x,y

def get_test_data(config,datas,labels):
    x = []
    y = []
    print "正在处理测试数据格式。"
    for data in datas:
        key_value = data.split(" ")
        # image = Image.open(key_value[0])
        # '''"判断是否为3通道"'''
        # if image.mode != "RGB":
        #     continue
        image = cv2.imread(key_value[0])
        image = cv2.resize(image, (config.normal_size, config.normal_size))
        image = img_to_array(image)
        x.append(image)
        # print key_value[1]
        if str(key_value[1]) in labels:
            # print labels.index(key_value[1])
            y.append(labels.index(key_value[1]))
        else:
            print "没有找到标签！"
    x = np.array(x, dtype="float") / 255.0
    y = np_utils.to_categorical(y, config.classes)
    # y = myone_hot(np.array(y))
    # y = one_hot(np.array(y), config.classes,1,0)
    return x,y


if __name__ =="__main__":
    '''先将需要读的文件全部打开'''
    # file_test = open("test_set.txt", "r")
    file_train = open("train_set.txt", "r")
    file_label = open("lable.txt", "r")
    labels = file_label.readlines()
    train_datas = file_train.readlines()
    # test_datas = file_test.readlines()
    '''获取样本数量'''
    num_train = len(train_datas)

    model = build_AlexNet(config)
    # x_test,y_test = get_test_data(config,test_datas,labels)
    for i in range(config.epochs):
        current_line = 0
        print "正在进行第"+str(i)+"轮训练！"
        while(current_line<num_train):
            x, y = get_train_data(config, current_line,train_datas,labels)
            # train(model,config,x,y,x_test,y_test)
            log = model.train_on_batch(x,y)
            print log
            # print x,y
            # print len(x),len(y)
            # model.fit(x.reshape(-1,config.normal_size,config.normal_size,config.channels), y, epochs=1, batch_size=config.batch_size,validation_split=0.8)
            # model.fit(x, y, epochs=1, batch_size=config.batch_size,validation_split=0.2)
            # loss,acc = model.evaluate(x_test,y_test,batch_size=config.batch_size)
            current_line = current_line+config.batch_size+1
        model.save(config.model_path+config.model_name)
    print "正在关闭文件。"
    # file_test.close()
    file_train.close()
    file_label.close()


    # print current_line,len(x),len(y)
    #
    # current_line = current_line+config.batch_size
    # x, y = get_train_data(config, current_line)
    # print current_line, len(x), len(y)
    #
    # current_line = current_line + config.batch_size
    # x, y = get_train_data(config, current_line)
    # print current_line, len(x), len(y)








