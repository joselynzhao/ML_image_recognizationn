#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:config.py
@TIME:2018/12/5 15:45
@DES:
'''

class DefaultConfig(object):
    cut_rate = 0.8
    path_root = "../data/101_ObjectCategories"
    normal_size = 227  # 图像输入网络之前需要被resize的大小

    weights_path = "../log/"  # 模型保存路径
    model_path ="../out_models/"

    channels = 3  # RGB通道数

    epochs = 1  # 训练的epoch次数
    batch_size = 512  # 训练的batch 数
    classes = 102  # 要识别的类数
    data_augmentation = False  # 是否使用keras的数据增强模块
    model_name = "test02"  # 选择所要使用的网络结构名称
    evaluate_model_path = "../models_for_test/"

config = DefaultConfig()
