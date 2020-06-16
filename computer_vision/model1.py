import os
import sys
import random
import numpy as np
from PIL import Image
from multiprocessing import Pool

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Activation # 레이어 추가
from tensorflow.keras import activations,optimizers,metrics #케라스 자체로만 하면 최신 버전 사용 가능
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout

class Data:
    def __init__(self):
        pass

    def get_path(self):
        img_list = [ (i[0],i[2]) for i in list(os.walk('C:\\Users\\TJ\\Desktop\\dyffo\\Python\\project_03\\data\\crl_image\\crl_image_resize'))[1:]]
        # for i in img_list:
        #     print(i[0],i[1],sep='\n')
        return img_list # [ ( str, list ) , ... ]

    def seperate_data(self,img_path):
        train = []
        test = []

        for path,file_list in img_path:
            lens = len(file_list)
            random.seed(1000)
            random.shuffle(file_list)
            point = int(0.7*lens)
            train.append((path,file_list[:point]))
            test.append((path,file_list[point:]))
        return train,test

    def get_img(self,tuples):
            path = tuples[0]
            file_list = tuples[1]

            data_x = [] # content
            data_y = [] # label

            label = path.split('\\')[-1]
            print(label)
            for file in file_list:
                filepath = '\\'.join((path,file))
                img_file = Image.open(filepath)
                img_array = np.array(img_file)
                data_x.append(img_array)
                data_y.append(label)

            return (data_x,data_y)



if __name__ == '__main__':

    data = Data()

    img_path = data.get_path()

    train,test = data.seperate_data(img_path)

    with Pool(processes=8) as pool:
        train_img = pool.map(data.get_img,train)
        # test_img = pool.map(data.get_img,test)

    # train_x = []
    # [train_x.extend(i[0]) for i in train_img] #input
    # train_x = np.array(train_x)
    # print(train_x.nbytes)
    # np.save('../../data/computer_vision_data/train_x.npy',train_x)

    # train_y = [i[1] for i in train_img] #output
    # with open('../../data/computer_vision_data/train_y.txt','w',encoding='utf-8') as f:
    #     f.write(','.join(train_y))
    # test_x = [i[0] for i in test_img] #input
    # np.save('../../data/computer_vision_data/test_ary.npy', test_x)
    # test_y = [i[1] for i in test_img] #output
    # with open('../../data/computer_vision_data/test_y.txt','w',encoding='utf-8') as f:
    #     f.write(','.join(test_y))

