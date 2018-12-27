#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:pre_handle.py
@TIME:2018/12/6 15:11
@DES:
'''

import  os
from scipy.misc import imread
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from config import config
import  sys




if __name__ =='__main__' :
    print sys.path
    type_list= os.listdir(config.path_root)
    k = 0
    type_value = set()
    while(k<1):
    # while(k<len(type_list)):
        path_typedir = config.path_root+'/'+type_list[k]
        image_list = os.listdir(path_typedir)  #得到了各个类别的书评数量

        # 转为了矩阵。。。全是正整数
        # pic  = imread("101_ObjectCategories/"+type_list[k]+'/'+image_list[0])
        # pic = mpimg.imread("101_ObjectCategories/"+type_list[k]+'/'+image_list[0])
        path_img = config.path_root+"/"+type_list[k]+'/'+image_list[0]
        img = Image.open(path_img)
        img = img.resize((64, 64), Image.ANTIALIAS)
        # img.show()  # 用预览工具打开
        img = np.array(img)




        '''利用plt绘图'''
        plt.figure("dog")
        plt.imshow(img)
        plt.axis('off')#是否要加坐标轴
        plt.show()

        print img.shape
        print img.dtype
        print img.size
        print type(img)
        ''' 输出结果如下
        (188, 300, 3)
        uint8
        169200
        <type 'numpy.ndarray'>
        '''


        # print img[2,3,2]

        '''
        220 被访问到的像素值
        '''

        '''
        随机生成5000个椒盐
        '''
        # rows, cols, dims = img.shape
        # for i in range(5000):
        #     x = np.random.randint(0, rows)
        #     y = np.random.randint(0, cols)
        #     img[x, y, :] = 255

        # plt.figure("beauty")
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()


        '''
        将lena图像二值化，像素值大于128的变为1，否则变为0
        '''
        # img =  np.array(Image.open(path_img).convert('L'))
        # rows, cols = img.shape
        # for i in range(rows):
        #     for j in range(cols):
        #         if (img[i, j] <= 128):
        #             img[i, j] = 0
        #         else:
        #             img[i, j] = 1
        #
        #
        # plt.figure("lena")
        # plt.imshow(img)
        # # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        #
        # print img.shape
        # print img.dtype
        # print img.size
        # print type(img)
        '''输出结果如下
        (188, 300)
        uint8
        56400
        <type 'numpy.ndarray'>
        '''


        '''测试输出单个图层'''
        # img2 = img[:,:,1]
        # plt.figure("beauty")
        # plt.imshow(img2)
        # plt.axis('off')
        # plt.show()
        #
        # img2 = img[:, :, 2]
        # plt.figure("beauty")
        # plt.imshow(img2)
        # plt.axis('off')
        # plt.show()
        #
        # img2 = img[:, :, 0]
        # plt.figure("beauty")
        # plt.imshow(img2)
        # plt.axis('off')
        # plt.show()

        k = k+1

         # type_value.add(type_list[k])
     # k = k+1
    # print len(type_list)
    # print type_value