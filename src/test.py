import os
import glob
import tarfile
import numpy as np
from scipy.io import loadmat
from shutil import copyfile, rmtree
import sys
import json

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

    print('sys version')

data_path = 'flower'
def download_file(url, dest=None):
    if not dest:
        dest = os.path.join(data_path, url.split('/')[-1])
    urlretrieve(url, dest)

    print ('download file')


def move_files(dir_name,cwd,labels):
    cur_dir_path = os.path.join(cwd, dir_name)
    if not os.path.exists(cur_dir_path):
        os.mkdir(cur_dir_path)
    for i in range(0, 102):
        class_dir = os.path.join(cwd, dir_name, str(i))
        os.mkdir(class_dir)
    for label in labels:
        src = str(label[0])
        dst = os.path.join(cwd,dir_name, label[1], src.split(os.sep)[-1])
        copyfile(src, dst)



def save_dict(content,filename):
    content = dict(content)
    with open(filename,'w') as file_object:
        json.dump(content,file_object)



def load_dict(filename):
    with open(filename,'r') as file_object:
        content = json.load(file_object)
    return content

def main():


    if not os.path.exists(data_path):
        os.mkdir(data_path)
    flowers_archive_path = os.path.join(data_path, '102flowers.tgz')
    if not os.path.isfile(flowers_archive_path):
        print ('Downloading images...')
        download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')

        tarfile.open(flowers_archive_path).extractall(path=data_path)
    image_labels_path = os.path.join(data_path, 'imagelabels.mat')
    if not os.path.isfile(image_labels_path):
        print("Downloading image labels...")
        download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')

    image_labels = loadmat(image_labels_path)['labels'][0]

    image_labels -= 1
    files = sorted(glob.glob(os.path.join(data_path, 'jpg', '*.jpg')))
    labels = np.array([i for i in zip(files, image_labels)])

    cwd = os.getcwd()

    dir_name = os.path.join(data_path, 'class')
    move_files(dir_name, cwd, labels)

    save_dict(labels, os.path.join(data_path, 'image-label.json'))



if __name__ == '__main__':
    main()