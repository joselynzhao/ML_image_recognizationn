#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:Evaluate.py
@TIME:2018/12/16 21:08
@DES:
'''
from main01 import *
from keras.models import load_model

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

if __name__ =="__main__":
    '''先将需要读的文件全部打开'''
    file_test = open("test_set.txt", "r")
    file_label = open("lable.txt", "r")
    labels = file_label.readlines()
    test_datas = file_test.readlines()
    x_test, y_test = get_test_data(config, test_datas, labels)
    model_list = os.listdir("../models_for_test/")
    for model_name in model_list:
        model = get_model("../models_for_test/"+model_name)
    loss,acc = model.evaluate(x_test,y_test,batch_size=config.batch_size)
    print loss,acc
    print "正在关闭文件。"
    file_test.close()
    file_label.close()
    # model = get_model(config.model_path + config.model_name)