#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:train01.py
@TIME:2018/12/16 20:58
@DES:
'''


from main01 import *


def train01(model,config,train_x,train_y):
    lr_reducer = ReduceLROnPlateau(factor=0.005, cooldown=0, patience=5, min_lr=0.5e-6,verbose=1)      #设置学习率衰减
    early_stopper = EarlyStopping(min_delta=0.001, patience=10,verbose=1)                                     #设置早停参数
    checkpoint = ModelCheckpoint(config.weights_path + config.model_name + "_model.h5",
                                 monitor="val_acc", verbose=1,
                                 save_best_only=True, save_weights_only=True,mode="max")            #保存训练过程中，在验证集上效果最好的模型
    model.fit(train_x, train_y, batch_size=config.batch_size,
              nb_epoch=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, checkpoint]
              )

def build_AlexNet01(config):
    model = Sequential()
    input_shape = (config.normal_size, config.normal_size, config.channels)
    model.add(Convolution2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(momentum=0.99,epsilon=0.001))
    model.add(
        Convolution2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

    model.add(
        Convolution2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # model.add(
    #     Convolution2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    print( "model is ready！")
    return model


if __name__ =="__main__":
    build_AlexNet01(config)
    # '''先将需要读的文件全部打开'''
    # # file_test = open("test_set.txt", "r")
    # file_train = open("train_set.txt", "r")
    # file_label = open("lable.txt", "r")
    # labels = file_label.readlines()
    # train_datas = file_train.readlines()
    # # test_datas = file_test.readlines()
    # '''获取样本数量'''
    # num_train = len(train_datas)
    #
    # model = build_AlexNet(config)
    # # x_test,y_test = get_test_data(config,test_datas,labels)
    # for i in range(config.epochs):
    #     current_line = 0
    #     print "正在进行第"+str(i)+"轮训练！"
    #     while(current_line<num_train):
    #         x, y = get_train_data(config, current_line,train_datas,labels)
    #         train01(model,config,x,y)
    #         # print x,y
    #         # print len(x),len(y)
    #         # model.fit(x.reshape(-1,config.normal_size,config.normal_size,config.channels), y, epochs=1, batch_size=config.batch_size,validation_split=0.8)
    #         # model.fit(x, y, epochs=1, batch_size=config.batch_size,validation_split=0.2)
    #         # loss,acc = model.evaluate(x_test,y_test,batch_size=config.batch_size)
    #         current_line = current_line+config.batch_size+1
    #     model.save(config.model_path+config.model_name)
    # print "正在关闭文件。"
    # # file_test.close()
    # file_train.close()
    # file_label.close()