#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:python_drew.py
@TIME:2018/12/18 22:18
@DES:
'''

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import  os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



def draw(x, plot_x,plot_y,lable,path,a_l):
    plt.title('Contrast training results')
    plt.plot(x,plot_x,linewidth=1,c='blue',label='normal')
    plt.plot(x,plot_y,linewidth=1,c='red',label='trash')
    #设置，x,y坐标标签和它的大小
    plt.xlabel("Batchs", fontsize=16)
    plt.ylabel(a_l, fontsize=16)
    #设置刻度数字的大小
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(lable, loc=0, ncol=2)

    print(path+'.png')
    plt.savefig(path+'.png')
    plt.show()
    plt.close()


def get_data(model_name):
    file_list = os.listdir(path_root+model_name)
    file_list.sort()
    datas=[]
    file = open(path_root+model_name+'/'+"data.txt","r")
    output = file.readlines()
    print len(output)
    for line in output:
        info = line.split(":")
        if len(info)==2:
            seg = info[1].split(",")
        else:
            continue
        s = seg[0].split("值")
        number = s[1]
        print number
        # break
        # segs=info.split("值")
        # if segs[1][-1]==',' or segs[1][-1]=='，':
        # print segs[1]
        # number = segs[1].strip(",")
        # print number
        # if segs[1]==0 or segs[1]==1:
        #     number = segs[1][:len(segs[1])-1].strip()
        # else:
        #     continue
            # print number
        # num = float(number)

        datas.append(number)
    datas = np.array(datas,dtype="float")
    return datas

def get_data0(model_name):
    file_list = os.listdir(path_root+model_name)
    file_list.sort()
    datas=[]
    file = open(path_root+model_name+'/'+"data.txt","r")
    output = file.readlines()
    print len(output)
    for line in output:
        info = line.split(" ")
        info = info[-1]
        s = info.split("率")
        number = s[1].strip()
        print number
        datas.append(number)
    datas = np.array(datas,dtype="float")
    return datas



path_root = "../terminal_out/"

if __name__ == "__main__":
    y=[]
    label=["model_normal","modelB64"]
    for la in label:
        model_name = la
        data = get_data(model_name)
        y.append(data)
    min_len = len(y[0])
    if min_len > len(y[1]):
        min_len = len(y[1])
        y[0]=y[0][:min_len]
    else:
        y[1]=y[1][:min_len]
    x = range(min_len)
    path = path_root+"1_1"
    draw(x,y[0],y[1],label,path,"Loss")

    # print len(y[0]),len(y[1])









    # x=range(7)
    # y1 = [1,2,3,4,5,4,5]
    # y2 = [3,2,2,4,5,4,3]
    # label=["line1","line2"]
    # draw(x,y1,y2,label)
    # draw2()